from collections import namedtuple, defaultdict
from copy import deepcopy
import numpy as np

Edge = namedtuple('Edge', (
    'x_from', 'y_from',
    'x_to', 'y_to',
    'weight',
))


def __get_weight_from_coordinates(x, y, reward_params, processed_state):
    assert len(processed_state.shape) == 3, 'the shape must be: [N_MASKS, MAP_SIZE, MAP_SIZE]'
    _, preys_mask, enemies_mask, bonuses_mask, _ = processed_state

    if preys_mask[y, x] == 1:
        return reward_params["w_kill_prey"]
    if enemies_mask[y, x] == 1:
        return reward_params["w_kill_enemy"]
    if bonuses_mask[y, x] == 1:
        return reward_params["w_kill_bonus"]

    return reward_params["w_dummy_step_start"]


def __get_adjacent_cells(x, y, reward_params, processed_state, can_pass_edges=True):
    assert len(processed_state.shape) == 3, 'the shape must be: [N_MASKS, MAP_SIZE, MAP_SIZE]'
    stones_mask = processed_state[0, ...]
    n, m = stones_mask.shape
    out = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        if can_pass_edges:
            nx = (x + dx) % m if x + dx >= 0 else m - 1
            ny = (y + dy) % n if y + dy >= 0 else n - 1
        else:
            nx = (x + dx)
            ny = (y + dy)
            if not 0 <= nx < m or not 0 <= ny < n:
                continue

        if stones_mask[ny, nx] != 1:
            out.append((nx, ny,
                        __get_weight_from_coordinates(nx, ny, reward_params, processed_state)))
    return out


def create_graph(reward_params, processed_state, can_pass_edges=True):
    assert len(processed_state.shape) == 3, 'the shape must be: [N_MASKS, MAP_SIZE, MAP_SIZE]'
    stones_mask = processed_state[0, ...]
    graph = []
    for y in range(stones_mask.shape[0]):
        for x in range(stones_mask.shape[1]):
            for x2, y2, w in __get_adjacent_cells(x, y, reward_params, processed_state, can_pass_edges):
                graph.append(Edge(x, y, x2, y2, w))
    return graph


def BellamanFord_modified(graph, reward_params, src, shape=(40, 40)):
    """kostyl in for loop"""
    D = [[float('-Inf')] * shape[1] for _ in range(shape[0])]  # distances
    D[src[1]][src[0]] = 0  # initialize distance of source as 0

    Used = defaultdict(set)

    for dummy_step_weight in np.linspace(
        reward_params['w_dummy_step_start'], reward_params['w_dummy_step_end'], reward_params['split_step']):
        D_tmp = deepcopy(D)
        for e in graph:
            weight = dummy_step_weight if (e.x_to, e.y_to) in Used[(e.x_from, e.y_from)] else e.weight
            weight = dummy_step_weight if weight < 0 else weight

            if D[e.y_from][e.x_from] + weight > D_tmp[e.y_to][e.x_to]:
                D_tmp[e.y_to][e.x_to] = D[e.y_from][e.x_from] + weight

                Used[(e.x_to, e.y_to)] = Used[(e.x_from, e.y_from)].copy()

                if weight > 0:
                    Used[(e.x_to, e.y_to)].add((e.x_to, e.y_to))

        D = deepcopy(D_tmp)

    return np.array(D)


def get_bonus_counts(info):
    return np.array([p['bonus_count'] for p in info['predators']])


def get_kills(info, next_info):
    """Returns prey kills and enemy kills for each predator of team 0 during the step"""
    n = len(next_info['predators'])
    prey_team_id = next_info['preys'][0]['team']
    prey_kills = np.zeros(n)
    enemy_kills = np.zeros(n)

    for killed, killer in next_info['eaten'].items():
        if killer[0] != 0:
            continue

        if killed[0] == prey_team_id:
            prey_kills[killer[1]] = 1
        else:
            enemy_kills[killer[1]] = 1

    bonus_counts = get_bonus_counts(info)
    bonus_counts_next = get_bonus_counts(next_info)
    bonus_kills = (bonus_counts_next > bonus_counts).astype(int)

    return prey_kills, enemy_kills, bonus_kills


# def get_state_value(reward_weights, processed_state, info, n_steps=7, src=(20, 20)):
#     """Returns ndarray[N_PREDATORS] containing state value for each predator"""
#     assert len(processed_state.shape) == 4, 'the shape must be: [N_PREDATORS, N_MASKS, MAP_SIZE, MAP_SIZE]'
#     n_predators = processed_state.shape[0]
#     bonus_counts = get_bonus_counts(info)

#     out = []
#     for i in range(n_predators):
#         weights = reward_weights.copy()
#         weights["w_kill_bonus"] = weights["w_kill_bonus"] * (0.5 ** bonus_counts[i])
#         weights["w_kill_enemy"] = weights["w_kill_enemy"] if bonus_counts[i] > 0 else 0
#         graph = __create_graph(weights, processed_state[i, ...])

#         state_value = BellamanFord_modified(graph, n_steps, reward_weights, src).max()

#         out.append(state_value)
#     return np.array(out)

# =============================================================================
def get_state_value(reward_params, processed_state, info, src=(20, 20)):
    """Returns ndarray[N_PREDATORS] containing state value for each predator
    generally state value is the maximum dst calculated with modified BellamanFord algorithm
    But if it is zero, sv is calculated with another function"""
    assert len(processed_state.shape) == 4, 'the shape must be: [N_PREDATORS, N_MASKS, MAP_SIZE, MAP_SIZE]'
    n_predators = processed_state.shape[0]
    bonus_counts = get_bonus_counts(info)

    out = []
    for i in range(n_predators):
        params = reward_params.copy()
        params["w_kill_bonus"] = params["w_kill_bonus"] * (0.5 ** bonus_counts[i])
        params["w_kill_enemy"] = params["w_kill_enemy"] if bonus_counts[i] > 0 else 0

        graph = create_graph(params, processed_state[i, ...])

        state_value = BellamanFord_modified(graph, params, src).max()

        if state_value == 0:
            state_value = get_alternative_state_value(params, processed_state[i, ...])

        out.append(state_value)
    return np.array(out)


def get_reward_mask(processed_state, reward_params):
    assert len(processed_state.shape) == 3, 'the shape must be: [N_MASKS, MAP_SIZE, MAP_SIZE]'
    stones_mask, preys_mask, enemies_mask, bonuses_mask, _ = processed_state
    reward_mask = np.zeros_like(stones_mask)
    reward_mask[preys_mask == 1] = reward_params['w_kill_prey']
    reward_mask[enemies_mask == 1] = reward_params['w_kill_enemy']
    reward_mask[bonuses_mask == 1] = reward_params['w_kill_bonus']
    return reward_mask


def get_alternative_state_value(reward_params, processed_state):
    assert len(processed_state.shape) == 3, 'the shape must be: [N_MASKS, MAP_SIZE, MAP_SIZE]'
    *_, distance_mask = processed_state
    reward_mask = get_reward_mask(processed_state, reward_params)
    weighted_reward_mask = reward_mask * (1 - distance_mask)
    return np.sum(weighted_reward_mask)
# =============================================================================


class Reward:
    def __init__(self, n_predators, reward_params):
        w_kill_prey, w_kill_enemy, w_kill_bonus = reward_params['w_kill_prey'], reward_params['w_kill_enemy'], reward_params['w_kill_bonus']
        self.reward_weights = reward_params
        self.n_predators = n_predators
        self.kill_weights = np.array([w_kill_prey, w_kill_enemy, w_kill_bonus])

        self.kills = None
        self.result = None

    def __call__(self, processed_state, info, next_processed_state, next_info):
        prey_kills, enemy_kills, bonus_kills = get_kills(info, next_info)
        self.kills = np.array([prey_kills, enemy_kills, bonus_kills]).transpose()

        self.state_value = get_state_value(self.reward_weights, processed_state, info)
        self.next_state_value = get_state_value(self.reward_weights, next_processed_state, next_info)
        self.sv_difference = self.next_state_value - self.state_value

        # set sv_difference to 0 if killed
        killed_anybody = np.logical_or(prey_kills, enemy_kills, bonus_kills)
        cond = np.logical_and(killed_anybody == 1, self.sv_difference < 0)
        self.sv_difference[cond] = 0

        self.result = self.sv_difference + self.kills @ self.kill_weights

        return self.result

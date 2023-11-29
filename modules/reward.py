from queue import Queue
import numpy as np

# ====== Helper functions for Reward and calc_expected_reward ===================


def __get_adjacent_cells(x, y, obstacles_mask, distance_mask):
    """Yields adjacent cells to (x, y) that are not obstacles and have not been visited"""
    n, m = obstacles_mask.shape
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx = (x + dx) % m if x + dx >= 0 else m - 1
        ny = (y + dy) % n if y + dy >= 0 else n - 1
        if obstacles_mask[ny, nx] != 1 and np.isnan(distance_mask[ny, nx]):
            yield (nx, ny)


def __get_n_nearest_targets(processed_state, n, src):
    """Returns list of tuples (x, y, dst) of n nearest targets 
    or more if n nearest targets are bonuses or enemies"""
    obstacles_mask, preys_mask, enemies_mask, bonuses_mask, _ = processed_state

    out = []

    queue = Queue()
    queue.put(src)

    distance_mask = np.empty_like(obstacles_mask, dtype=np.float32)
    distance_mask.fill(np.nan)
    distance_mask[src[1], src[0]] = 0

    contains_preys = False

    while not queue.empty():
        x, y = queue.get()

        for nx, ny in __get_adjacent_cells(x, y, obstacles_mask, distance_mask):
            queue.put((nx, ny))
            distance_mask[ny, nx] = distance_mask[y, x] + 1

            if preys_mask[ny, nx] == 1:
                contains_preys = True
                out.append((nx, ny, distance_mask[ny, nx]))

            if (enemies_mask[ny, nx] == 1 or bonuses_mask[ny, nx] == 1) and len(out) < n:
                out.append((nx, ny, distance_mask[ny, nx]))

            if len(out) >= n and contains_preys:
                return out

    return out


def __get_weight_from_coordinates(x, y, reward_params, processed_state, bonus_count):
    _, preys_mask, enemies_mask, bonuses_mask, _ = processed_state

    if preys_mask[y, x] == 1:
        return reward_params["w_kill_prey"]
    if enemies_mask[y, x] == 1:
        return reward_params["w_kill_enemy"]
    if bonuses_mask[y, x] == 1:
        return reward_params["w_kill_bonus"] * (reward_params['gamma_for_bonus_count'] ** bonus_count)

    return 0


def __get_target_density(processed_state, x, y, reward_params, area=11):
    obstacles_mask, _, _, bonuses_mask, _ = processed_state
    penalties = np.linspace(1, 0, area)
    density = 0

    queue = Queue()
    queue.put((x, y))

    distance_mask = np.empty_like(obstacles_mask, dtype=np.float32)
    distance_mask.fill(np.nan)
    distance_mask[y, x] = 0

    curr_dst = 0
    bonus_count = 0
    while not queue.empty():
        x, y = queue.get()
        density += penalties[int(distance_mask[y, x])] * __get_weight_from_coordinates(x,
                                                                                       y, reward_params, processed_state, bonus_count)
        bonus_count += int(bonuses_mask[y, x])

        for nx, ny in __get_adjacent_cells(x, y, obstacles_mask, distance_mask):
            queue.put((nx, ny))
            distance_mask[ny, nx] = distance_mask[y, x] + 1
            curr_dst = max(curr_dst, distance_mask[ny, nx])

        if curr_dst >= area:
            return density

    return density


def __get_n_nearest_target_values(processed_state, reward_params, src):
    """Returns list of tuples (x, y, dst, target_value) of n nearest targets 
    or more if n nearest targets are bonuses or enemies"""
    n_nearest_targets = __get_n_nearest_targets(processed_state, reward_params['n_nearest_targets'], src)
    out = []

    for x, y, dst in n_nearest_targets:
        target_density = __get_target_density(processed_state, x, y, reward_params)
        target_value = target_density / dst
        out.append((x, y, dst, target_value))

    return out


def get_best_target_distance(processed_state, reward_params, src=(20, 20)):
    n_nearest_target_values = __get_n_nearest_target_values(processed_state, reward_params, src)
    highest_target_value = float('-inf')
    best_target_distance = None  # dst to target with highest value

    for *_, dst, target_value in n_nearest_target_values:
        if target_value > highest_target_value:
            highest_target_value = target_value
            best_target_distance = dst

    return best_target_distance

# ====== Expected reward (not used for training) ================================


def __get_expected_bonus_counts(processed_state, bonus_counts, dx, dy, src=(20, 20)):
    """helper function for calc_expected_reward"""
    expected_bonus_counts = bonus_counts.copy()

    for i, pr_st in enumerate(processed_state):
        _, _, enemies_mask, bonuses_mask, _ = pr_st

        nx, ny = src[0] + dx, src[1] + dy

        if enemies_mask[ny, nx] == 1:
            expected_bonus_counts[i] -= 1
        if bonuses_mask[ny, nx] == 1:
            expected_bonus_counts[i] += 1

    return expected_bonus_counts


def __get_expected_kills(processed_state, dx, dy, src=(20, 20)):
    """helper function for calc_expected_reward"""
    n_predators = processed_state.shape[0]
    expected_kills = np.zeros((n_predators, 3))

    for i, pr_st in enumerate(processed_state):
        _, preys_mask, enemies_mask, bonuses_mask, _ = pr_st

        nx, ny = src[0] + dx, src[1] + dy

        if preys_mask[ny, nx] == 1:
            expected_kills[i, 0] = 1
        if enemies_mask[ny, nx] == 1:
            expected_kills[i, 1] = 1
        if bonuses_mask[ny, nx] == 1:
            expected_kills[i, 2] = 1

    return expected_kills


def __get_actual_params(reward_params, bonus_count):
    """helper function for calc_expected_reward"""
    params = reward_params.copy()
    params["w_kill_bonus"] = params["w_kill_bonus"] * (params['gamma_for_bonus_count'] ** bonus_count)
    params["w_kill_enemy"] = params["w_kill_enemy"] if bonus_count > 0 else 0
    return params


def __get_expected_distances(processed_state, reward_params, bonus_counts, src=(20, 20), dx=0, dy=0):
    """helper function for calc_expected_reward"""
    result = []

    for i, pr_st in enumerate(processed_state):
        params = __get_actual_params(reward_params, bonus_counts[i])
        nx, ny = src[0] + dx, src[1] + dy
        result.append(get_best_target_distance(pr_st, params, src=(nx, ny)))

    result = [x if x is not None else np.nan for x in result]
    return np.array(result)


def __get_actual_kill_weights(bonus_counts, reward_params, n_predators):
    """helper function for calc_expected_reward"""
    kill_weights = np.empty((n_predators, 3), dtype=np.float32)

    for i in range(n_predators):
        params = __get_actual_params(reward_params, bonus_counts[i])
        kill_weights[i, 0] = params['w_kill_prey']
        kill_weights[i, 1] = params['w_kill_enemy']
        kill_weights[i, 2] = params['w_kill_bonus']

    return kill_weights


def __goes_to_obstacle(processed_state, dx, dy, src=(20, 20)):
    """helper function for calc_expected_reward"""
    goes_to_obstacle = np.zeros(processed_state.shape[0])

    for i, pr_st in enumerate(processed_state):
        obstacles_mask, *_ = pr_st

        nx, ny = src[0] + dx, src[1] + dy

        if obstacles_mask[ny, nx] == 1:
            goes_to_obstacle[i] = 1

    return goes_to_obstacle


def calc_expected_reward(processed_state, info, reward_params, dx, dy, max_dist_change=2):
    """This function can be used for RewardBasedModel dedicated to check quality of reward system"""
    bonus_counts = get_bonus_counts(info)
    next_bonus_counts = __get_expected_bonus_counts(processed_state, bonus_counts, dx, dy)
    distances = __get_expected_distances(processed_state, reward_params, bonus_counts)
    next_distances = __get_expected_distances(processed_state, reward_params, next_bonus_counts, dx=dx, dy=dy)

    # in some rare cases some of distances can be NaN
    isnan = np.logical_or(np.isnan(distances), np.isnan(next_distances))
    distances[isnan] = 0
    next_distances[isnan] = 0

    dist_difference = next_distances - distances

    excepted_kills = __get_expected_kills(processed_state, dx, dy)

    # set distances to 0 if killed
    killed_anybody = np.logical_or(excepted_kills[:, 0] == 1, excepted_kills[:, 1] == 1, excepted_kills[:, 2] == 1)
    dist_difference[killed_anybody == 1] = 0

    # set distances to 0 if sudden change
    sudden_change = np.logical_or(dist_difference > max_dist_change, dist_difference < -max_dist_change)
    dist_difference[sudden_change] = 0

    # set distances to 0 if goes to obstacle
    goes_to_obstacle = __goes_to_obstacle(processed_state, dx, dy)
    dist_difference[goes_to_obstacle == 1] = 0

    # clip to [-1, 1] because agent cannot control 1 -> 2 or -2 -> -1
    dist_difference = np.clip(dist_difference, -1, 1)

    kill_weights = __get_actual_kill_weights(bonus_counts, reward_params, processed_state.shape[0])

    result = dist_difference * reward_params['w_dist_change'] + \
        np.sum(excepted_kills * kill_weights, axis=1)

    return result

# ====== Reward =================================================================


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

def check_for_standing_still(info, next_info):
    out = []
    for predator_info, next_predator_info in zip(info['predators'], next_info['predators']):
        out.append(
            predator_info['x'] == next_predator_info['x'] and
            predator_info['y'] == next_predator_info['y']
        )
    return np.array(out)


class Reward:
    def __init__(self, global_config, train_config, max_dist_change=2):
        self.max_dist_change = max_dist_change
        self.n_predators = global_config.n_predators
        self.reward_params = train_config.reward_params

        self.dist_difference = None
        self.kills = None
        self.result = None

    def __call__(self, processed_state, info, next_processed_state, next_info):
        processed_state, bonus_counts = processed_state
        next_processed_state, next_bonus_counts = next_processed_state
        distances = self.__get_distances(processed_state, bonus_counts)
        next_distances = self.__get_distances(next_processed_state, next_bonus_counts)

        # in some rare cases some of distances can be NaN
        isnan = np.logical_or(np.isnan(distances), np.isnan(next_distances))
        distances[isnan] = 0
        next_distances[isnan] = 0

        self.dist_difference = next_distances - distances

        prey_kills, enemy_kills, bonus_kills = get_kills(info, next_info)
        self.kills = np.array([prey_kills, enemy_kills, bonus_kills]).transpose()

        # set distances to 0 if killed
        killed_anybody = np.logical_or(prey_kills, enemy_kills, bonus_kills)
        self.dist_difference[killed_anybody == 1] = 0

        # set distances to 0 if sudden change.
        # This means that state drastically changed without actions of the agent.
        # Example_1: Enemy ate the prey which was the closest one
        # Example_2: respawned
        sudden_change = np.logical_or(self.dist_difference > self.max_dist_change,
                                      self.dist_difference < -self.max_dist_change)
        self.dist_difference[sudden_change] = 0

        # clip to [-1, 1] because agent cannot control 1 -> 2 or -2 -> -1
        self.dist_difference = np.clip(self.dist_difference, -1, 1)

        kill_weights = self.__get_actual_kill_weights(bonus_counts)

        self.result = self.dist_difference * self.reward_params['w_dist_change'] + \
            np.sum(self.kills * kill_weights, axis=1)
        
        # punishment for standing still
        stands_still = check_for_standing_still(info, next_info)
        self.result[stands_still == 1] = self.reward_params['standing_still_penalty']

        return self.result

    def __get_distances(self, processed_state, bonus_counts):
        result = []

        for i, pr_st in enumerate(processed_state):
            params = self.__get_actual_params(bonus_counts[i])
            result.append(get_best_target_distance(pr_st, params))

        return np.array([x if x is not None else np.nan for x in result])

    def __get_actual_params(self, bonus_count):
        params = self.reward_params.copy()
        params["w_kill_bonus"] = params["w_kill_bonus"] * (params['gamma_for_bonus_count'] ** bonus_count)
        params["w_kill_enemy"] = params["w_kill_enemy"] if bonus_count > 0 else 0
        return params

    def __get_actual_kill_weights(self, bonus_counts):
        kill_weights = np.empty((self.n_predators, 3), dtype=np.float32)

        for i in range(self.n_predators):
            params = self.__get_actual_params(bonus_counts[i])
            kill_weights[i, 0] = params['w_kill_prey']
            kill_weights[i, 1] = params['w_kill_enemy']
            kill_weights[i, 2] = params['w_kill_bonus']

        return kill_weights


# ===============================================================================

class RewardBasedModel:
    def __init__(self, train_config):
        self.reward_params = train_config.reward_params        
        self.expected_info = dict()

    def get_actions(self, processed_state, info):
        expected_rewards = []
        for name, dx, dy in [("right", 1, 0), ("left", -1, 0), ("up", 0, -1), ("down", 0, 1)]:
            expected_reward = calc_expected_reward(processed_state, info, self.reward_params, dx, dy)
            expected_rewards.append(expected_reward)
            self.expected_info[name] = expected_reward
        self.expected_rewards = np.stack(expected_rewards).transpose()
        out = np.argmax(self.expected_rewards, axis=1) + 1
        return out

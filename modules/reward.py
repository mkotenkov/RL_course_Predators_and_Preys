from queue import Queue
import numpy as np

def get_kills(info):
    """Returns prey kills and enemy kills for each predator of team 0 during the step"""
    n = len(info['predators'])
    prey_team_id = info['preys'][0]['team']
    prey_kills = np.zeros(n)
    enemy_kills = np.zeros(n)

    for killed, killer in info['eaten'].items():
        if killer[0] != 0:
            continue

        if killed[0] == prey_team_id:
            prey_kills[killer[1]] = 1
        else:
            enemy_kills[killer[1]] = 1

    return prey_kills, enemy_kills


def get_adjacent_cells(x, y, stones_mask, distance_mask):
    """Yields adjacent cells to (x, y) that are not obstacles and have not been visited"""
    n, m = stones_mask.shape
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx = (x + dx) % m if x + dx >= 0 else m - 1
        ny = (y + dy) % n if y + dy >= 0 else n - 1                        
        if stones_mask[ny, nx] != 1 and np.isnan(distance_mask[ny, nx]):            
            yield (nx, ny)

def get_distances(processed_state, x, y, big_num=10):
    """Estimates distances to nearest (prey or enemy) and teammate from the cell(x, y)
    processed state here: [N_MASKS, MAP_SIZE, MAP_SIZE] with main teammate at (20, 20)"""

    stones_mask, preys_or_enemies_mask, teammates_mask = processed_state[
        0, ...], processed_state[1, ...], processed_state[2, ...]

    queue = Queue()
    queue.put((x, y))

    distance_mask = np.empty_like(stones_mask, dtype=np.float32)
    distance_mask.fill(np.nan)    
    distance_mask[y, x] = 0    

    d_prey_or_enemy = None
    d_teammate = None

    # BFS
    while not queue.empty():
        x, y = queue.get()        

        for nx, ny in get_adjacent_cells(x, y, stones_mask, distance_mask):
            queue.put((nx, ny))
            distance_mask[ny, nx] = distance_mask[y, x] + 1

            if preys_or_enemies_mask[ny, nx] == 1 and d_prey_or_enemy is None:
                d_prey_or_enemy = distance_mask[ny, nx]                

            if teammates_mask[ny, nx] == 1 and d_teammate is None:
                d_teammate = distance_mask[ny, nx]                

            if d_prey_or_enemy is not None and d_teammate is not None:
                return d_prey_or_enemy, d_teammate

    d_prey_or_enemy = big_num if d_prey_or_enemy is None else d_prey_or_enemy
    d_teammate = big_num if d_teammate is None else d_teammate

    return d_prey_or_enemy, d_teammate


class Reward:
    """combined state is matirx with {n_predators} rows and {2} cols: 
    [dist_prey_or_enemy, dist_teammate]"""

    def __init__(self, n_predators, w_d_prey_or_enemy, w_d_teammate,
                  w_kill_prey, w_kill_enemy, max_dist_change=2):        
        self.max_dist_change = max_dist_change
        self.n_predators = n_predators
        self.dist_weights = np.array([w_d_prey_or_enemy, w_d_teammate])
        self.kill_weights = np.array([w_kill_prey, w_kill_enemy])
        self.dist_difference = None
        self.kills = None

    def __call__(self, processed_state, next_processed_state, next_info):
        comb_state = self.__create_comb_state_matrix(processed_state)
        next_comb_state = self.__create_comb_state_matrix(next_processed_state)
        self.dist_difference = (next_comb_state - comb_state)

        prey_kills, enemy_kills = get_kills(next_info)
        self.kills = np.array([prey_kills, enemy_kills]).transpose()

        # set distances to 0 if killed
        killed_anybody = np.logical_or(prey_kills, enemy_kills)
        self.dist_difference[killed_anybody, :] = 0

        # set distances to 0 if sudden change. 
        # This means that state drastically changed without actions of the agent. 
        # Example_1: Enemy ate the prey which was the closest one
        # Example_2: respawned
        sudden_change = np.logical_or(self.dist_difference > self.max_dist_change,
                                      self.dist_difference < -self.max_dist_change)
        self.dist_difference[sudden_change] = 0

        return self.dist_difference @ self.dist_weights + self.kills @ self.kill_weights

    def __create_comb_state_matrix(self, processed_state):
        result = np.empty((self.n_predators, 2))
        for i, i_processed_state in enumerate(processed_state):
            d_prey_or_enemy, d_teammate = get_distances(i_processed_state, 20, 20)
            result[i, 0] = d_prey_or_enemy            
            result[i, 1] = d_teammate            
        return result

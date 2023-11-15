from queue import Queue
import numpy as np

# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# def paint_mask(mask):
#     clear_output(wait=True)
#     plt.imshow(mask, cmap="gray")
#     plt.axis("off")
#     plt.show()


def get_adjacent_cells(x, y, stones_mask, distance_mask):
    """Yields adjacent cells to (x, y) that are not obstacles and have not been visited"""
    n, m = stones_mask.shape
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx = (x + dx) % m if x + dx >= 0 else m - 1
        ny = (y + dy) % n if y + dy >= 0 else n - 1
        if stones_mask[ny, nx] != 1 and np.isnan(distance_mask[ny, nx]):
            yield (nx, ny)


def get_distance_mask(centered_stones_mask, source=(20, 20)):
    queue = Queue()
    queue.put(source)

    distance_mask = np.empty_like(centered_stones_mask, dtype=np.float32)
    distance_mask.fill(np.nan)
    distance_mask[source[1], source[0]] = 0

    while not queue.empty():
        x, y = queue.get()

        for nx, ny in get_adjacent_cells(x, y, centered_stones_mask, distance_mask):
            queue.put((nx, ny))
            distance_mask[ny, nx] = distance_mask[y, x] + 1

    distance_mask[np.isnan(distance_mask)] = -1
    distance_mask = distance_mask / distance_mask.max()
    distance_mask[distance_mask < 0] = 2
    return distance_mask


def preprocess(state, info):
    num_teams = info['preys'][0]['team']

    stones_mask = np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == -1).astype(np.float64)
    preys_mask = (state[:, :, 0] == num_teams).astype(np.float64)
    enemies_mask = np.logical_and(state[:, :, 0] > 0, state[:, :, 0] < num_teams).astype(np.float64)
    bonuses_mask = np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == 1).astype(np.float64)
    # preys_or_enemies_mask = np.logical_or(preys_mask, enemies_mask)
    # teammates_mask = state[:, :, 0] == 0

    coords = [(predator['x'], predator['y']) for predator in info['predators']]

    n, m, _ = state.shape
    vertical_center = n // 2
    horizontal_center = m // 2

    output = []
    for x, y in coords:
        bias = (horizontal_center - x, vertical_center - y)

        centered_stones_mask = np.roll(stones_mask, bias, axis=(1, 0))
        distance_mask = get_distance_mask(centered_stones_mask)

        output.append(np.stack([
            centered_stones_mask,
            np.roll(preys_mask, bias, axis=(1, 0)),
            np.roll(enemies_mask, bias, axis=(1, 0)),
            # np.roll(teammates_mask, bias, axis=(1, 0)),
            np.roll(bonuses_mask, bias, axis=(1, 0)),
            distance_mask
        ]))

    return np.stack(output)

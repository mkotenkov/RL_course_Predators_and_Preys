import numpy as np


def preprocess(state, info):
    num_teams = info['preys'][0]['team']

    stones_mask = np.logical_and(state[:, :, 0] == -1, state[:, :, 1] == -1)

    enemies_mask = np.logical_and(state[:, :, 0] > 0, state[:, :, 0] < num_teams).astype(np.float64)
    preys_mask = state[:, :, 0] == num_teams
    preys_or_enemies_mask = np.logical_or(preys_mask, enemies_mask)

    teammates_mask = state[:, :, 0] == 0

    coords = [(predator['x'], predator['y']) for predator in info['predators']]

    n, m, _ = state.shape
    vertical_center = n // 2
    horizontal_center = m // 2

    output = []
    for x, y in coords:
        bias = (horizontal_center - x, vertical_center - y)
        output.append(np.stack([
            np.roll(stones_mask, bias, axis=(1, 0)),
            np.roll(preys_or_enemies_mask, bias, axis=(1, 0)),
            np.roll(teammates_mask, bias, axis=(1, 0))
        ]))

    return np.stack(output)

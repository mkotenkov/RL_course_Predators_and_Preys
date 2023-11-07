from world.envs import OnePlayerEnv, VersusBotEnv
from world.realm import Realm
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamMapLoader, SingleTeamRocksMapLoader
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import ClosestTargetAgent, Dummy
from world.utils import RenderedEnvWrapper

from collections import defaultdict, namedtuple
from IPython.display import clear_output
from dataclasses import dataclass

import os
from matplotlib import pyplot as plt

import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

@dataclass
class TrainConfig:
    description: str
    max_steps_for_episode: int
    gamma: float
    initial_steps: int
    steps: int
    steps_per_update: int
    steps_per_paint: int
    steps_per_gif: int
    buffer_size: int
    batch_size: int
    learning_rate: float
    eps_start: float
    eps_end: float
    eps_decay: int
    tau: float
    reward_weights: dict
    seed: int

    def __str__(self):
        return '\n'.join([f'{k}: {v}' for k, v in vars(self).items()])


class Logger:
    def __init__(self, config, path_to_general_folder='logs'):
        self.data = defaultdict(list)
        self.config = config
        self.path_to_general_folder = path_to_general_folder

        os.makedirs(self.path_to_general_folder, exist_ok=True)

        subfolders_names_ints = [
            int(y) for y in [x[1] for x in os.walk(self.path_to_general_folder)][0]]
        curr_subfolder_name = str(max(subfolders_names_ints) + 1) if len(
            subfolders_names_ints) > 0 else '1'

        self.curr_subfolder_path = f'{self.path_to_general_folder}/{curr_subfolder_name}'
        os.makedirs(self.curr_subfolder_path)

    def add(self, key, value):
        self.data[key].append(value)

    def save(self):
        for k, v in self.data.items():
            np.save(f'{self.curr_subfolder_path}/{k}.npy', np.array(v))
        with open(f'{self.curr_subfolder_path}/config.txt', 'w') as f:
            f.write(str(self.config))

    @classmethod
    def load(cls, path_to_folder):
        self = cls.__new__(cls)

        # restore paths
        self.curr_subfolder_path = path_to_folder
        self.path_to_general_folder = '/'.join(path_to_folder.split('/')[:-1])

        # restore data
        self.data = defaultdict(list)
        for filename in os.listdir(path_to_folder):
            if filename.endswith('.npy'):
                k = filename.split('.')[0]
                self.data[k] = np.load(f'{path_to_folder}/{filename}').tolist()

        # restore config
        with open(f'{path_to_folder}/config.txt', 'r') as f:
            d = dict()
            for line in f.read().split('\n'):
                k, v = line.split(': ', maxsplit=1)
                try:
                    d[k] = eval(v)
                except Exception:
                    d[k] = v
        self.config = TrainConfig(**d)

        return self


def paint(logger, groups):
    clear_output(wait=True)

    for group in groups:
        for k, v in logger.data.items():
            if k in group:                
                plt.plot(v, label=k)
        plt.legend()
        plt.show()

def get_env(n_predators, steps_done, step_limit, render_gif=False):
    base = VersusBotEnv(Realm(
        map_loader=TwoTeamMapLoader(),
        playable_teams_num=2,
        playable_team_size=n_predators,
        bots={1: ClosestTargetAgent()},
        step_limit=step_limit
    ))
    return RenderedEnvWrapper(base) if render_gif else base


    

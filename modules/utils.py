from world.envs import OnePlayerEnv, VersusBotEnv
from world.realm import Realm
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamMapLoader, SingleTeamRocksMapLoader
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import ClosestTargetAgent, Dummy
from world.utils import RenderedEnvWrapper

from modules.create_nice_gif import create_gif, get_text_info, create_video_from_gif
from modules.preprocess import preprocess
from modules.reward import Reward


import os
import random
import numpy as np

from IPython.display import clear_output
from dataclasses import dataclass
from matplotlib import pyplot as plt
from collections import defaultdict


@dataclass
class TrainConfig:
    description: str
    max_steps_for_episode: int
    gamma: float
    initial_steps: int
    steps: int
    steps_per_update: int
    steps_per_paint: int
    steps_per_eval: int
    buffer_size: int
    batch_size: int
    learning_rate: float
    eps_start: float
    eps_end: float
    eps_decay: int
    tau: float
    reward_params: dict
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

# def get_env(n_predators, difficulty, step_limit, render_gif=False):
#     assert 0 <= difficulty <= 1
#     base = VersusBotEnv(Realm(
#         map_loader=TwoTeamMapLoader(),
#         playable_teams_num=2,
#         playable_team_size=n_predators,
#         bots={1: ClosestTargetAgent()},
#         step_limit=step_limit
#     ))
#     return RenderedEnvWrapper(base) if render_gif else base


def get_env(n_predators, difficulty, step_limit, render_gif=False):
    """
    -with difficulty from -1 to 0: returns map without rocks 
    and gradually increases number of moving preys

    -with difficulty from 0 to 1: returns Labyrinth or Rocks
    and gradually increases difficulty of map"""
    assert -1 <= difficulty <= 1

    if difficulty < 0:
        MapLoader = TwoTeamMapLoader
        kwargs_range = dict(
            difficulty=[0., 1.],
        )
        difficulty += 1
    else:
        if random.random() > 0.5:
            MapLoader = TwoTeamLabyrinthMapLoader
            kwargs_range = dict(
                additional_links_max=[24, 12],
                additional_links_min=[3, 1]
            )
        else:
            MapLoader = TwoTeamRocksMapLoader
            kwargs_range = dict(
                rock_spawn_proba=[0.01, 0.15],
                additional_rock_spawn_proba=[0.0, 0.21]
            )

    generation_kwargs = dict()
    for k, v in kwargs_range.items():
        value = v[0] + (v[1] - v[0]) * difficulty
        value = int(value) if MapLoader == TwoTeamLabyrinthMapLoader else value
        generation_kwargs[k] = value

    base = VersusBotEnv(Realm(
        map_loader=MapLoader(**generation_kwargs),
        playable_teams_num=2,
        playable_team_size=n_predators,
        bots={1: ClosestTargetAgent()},
        step_limit=step_limit
    ))
    return RenderedEnvWrapper(base) if render_gif else base


def simulate_episode(model, difficulty, n_predators, cfg, gif_path=None, render_gif=False):
    env = get_env(n_predators, difficulty, cfg.max_steps_for_episode, render_gif=render_gif)
    state, info = env.reset()
    processed_state = preprocess(state, info)
    done = False
    r = Reward(n_predators, cfg.reward_params)
    text_info = [get_text_info(r, info, env, model)]
        
    while not done:    
        actions = model.get_actions(processed_state, info)
        next_state, done, next_info = env.step(actions)
        next_processed_state = preprocess(next_state, next_info)
        reward = r(processed_state, info, next_processed_state, next_info)
        info, processed_state = next_info, next_processed_state
        text_info.append(get_text_info(r, next_info, env, model))  # for display        

    if render_gif and gif_path is not None:
        create_gif(env, gif_path, duration=1., text_info=text_info)
        create_video_from_gif(gif_path)

    sum_ = (info['scores'][0] + info['scores'][1])
    return (info['scores'][0] / sum_) if sum_ > 0 else None

def evaluate(model, n_predators, cfg, n_episodes=5):
    results = []
    for d in np.linspace(0, 1, n_episodes):              
        results.append(simulate_episode(model, d, n_predators, cfg))
    return sum(results) / len(results)

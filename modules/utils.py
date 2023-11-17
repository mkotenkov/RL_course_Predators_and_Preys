from world.envs import VersusBotEnv
from world.realm import Realm
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import ClosestTargetAgent
from world.utils import RenderedEnvWrapper

from modules.create_gif import create_gif, get_text_info, create_video_from_gif
from modules.preprocess import preprocess
from modules.reward import Reward

import os
import random
import numpy as np

from IPython.display import clear_output
from dataclasses import dataclass
from matplotlib import pyplot as plt
from collections import defaultdict
from copy import deepcopy


@dataclass
class GlobalConfig:
    device: str
    n_actions: int
    n_predators: int
    n_masks: int
    map_size: int


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
    def __init__(self, config, model, path_to_general_folder='logs'):
        self.data = defaultdict(list)
        self.config = config
        self.model = model
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
        with open(f'{self.curr_subfolder_path}/architecture.txt', 'w') as f:
            f.write(str(self.model.dqn))

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
                self.data[k] = np.load(f'{path_to_folder}/{filename}', allow_pickle=True).tolist()

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


def __smooth_array(arr, window_size):
    size = len(arr) - 1 if len(arr) < window_size else window_size
    return np.convolve(arr, np.ones(size) / size, mode='valid')

def paint(logger, save_plots=False, reward_window=1000, loss_window=100):
    clear_output(wait=True)

    n = len(logger.data['reward'])

    smoothed_reward_per_step = __smooth_array(logger.data['reward'], min(n, reward_window))
    smoothed_loss = __smooth_array(logger.data['loss'], min(n, loss_window))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(smoothed_reward_per_step, label=f'smoothed_reward_per_step ({reward_window})')
    plt.title('Reward')
    plt.xlabel('Steps')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(smoothed_loss, label=f'smoothed_loss ({loss_window})')
    plt.xlabel('Steps')
    plt.title('Loss')
    plt.legend()
    plt.grid()

    if save_plots:
        plt.savefig(logger.curr_subfolder_path + '/reward_loss.png')

    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(logger.data['score_difference'], label='score_difference')
    plt.xlabel('Steps')
    plt.title('Avg of 5 episodes: \n score / (bot_score + score)')
    plt.axhline(0.5, color='red', linestyle='--')
    plt.legend()
    plt.grid()

    if save_plots:
        plt.savefig(logger.curr_subfolder_path + '/score_difference.png')

    plt.show()

def get_env(global_config, train_config, difficulty, render_gif=False):
    assert 0 <= difficulty <= 1

    plain_map_proba = 1 - difficulty if difficulty > 0.15 else 1
    probs = [plain_map_proba, (1 - plain_map_proba) / 2, (1 - plain_map_proba) / 2]
    choice = np.random.choice(3, 1, p=probs)[0]

    if choice == 0:
        MapLoader = TwoTeamMapLoader
        kwargs_range = dict(
            move_proba=[0., 1.],
        )        

    elif choice == 1:
        MapLoader = TwoTeamLabyrinthMapLoader
        kwargs_range = dict(
            additional_links_max=[24, 12],
            additional_links_min=[3, 1],
            move_proba=[0., 1.],
        )

    else:
        MapLoader = TwoTeamRocksMapLoader
        kwargs_range = dict(
            rock_spawn_proba=[0.01, 0.15],
            additional_rock_spawn_proba=[0.0, 0.21],
            move_proba=[0., 1.],
        )

    generation_kwargs = dict()
    for k, v in kwargs_range.items():
        value = v[0] + (v[1] - v[0]) * difficulty
        value = int(value) if MapLoader == TwoTeamLabyrinthMapLoader else value
        generation_kwargs[k] = value

    base = VersusBotEnv(Realm(
        map_loader=MapLoader(**generation_kwargs),
        playable_teams_num=2,
        playable_team_size=global_config.n_predators,
        bots={1: ClosestTargetAgent()},
        step_limit=train_config.max_steps_for_episode
    ))
    return RenderedEnvWrapper(base) if render_gif else base


def simulate_episode(model, difficulty, gif_path=None):
    global_config = model.global_config
    train_config = model.train_config

    render_gif = gif_path is not None

    env = get_env(global_config, train_config, difficulty, render_gif=render_gif)
    state, info = env.reset()
    processed_state = preprocess(state, info)
    done = False
    r = Reward(global_config, train_config)
    # getting actions here (not as the first step of wile loop) to display q_values in gif before the action is done
    # actions = model.get_actions(processed_state)    
    text_info = [get_text_info(r, info, env, model)]

    while not done:      
        prev_model = deepcopy(model) 
        actions = model.get_actions(processed_state)  
        next_state, done, next_info = env.step(actions)
        next_processed_state = preprocess(next_state, next_info)
        _ = r(processed_state, info, next_processed_state, next_info)
        info, processed_state = next_info, next_processed_state
        text_info.append(get_text_info(r, next_info, env, prev_model))  # for display
           

    if render_gif:
        create_gif(env, gif_path, duration=1., text_info=text_info)
        create_video_from_gif(gif_path)

    sum_ = (info['scores'][0] + info['scores'][1])
    return (info['scores'][0] / sum_) if sum_ > 0 else None


def evaluate(model, n_episodes=10):
    results = []
    for d in np.linspace(-0.5, 1, n_episodes):
        results.append(simulate_episode(model, d))
    return sum(results) / len(results)

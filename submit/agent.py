import torch
from dataclasses import dataclass

from .DQN import DQN
from .preprocess import preprocess

# =================================================================================================


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


global_config = GlobalConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    n_actions=5,
    n_predators=5,
    n_masks=5,
    map_size=40
)

train_config = TrainConfig(
    description='inference...',
    max_steps_for_episode=300,
    gamma=0.9,
    initial_steps=300,  # 3000
    steps=100_000,
    steps_per_update=3,
    steps_per_paint=250,  # 500
    steps_per_eval=1000,  # 5000
    buffer_size=10_000,
    batch_size=64,
    learning_rate=0.001,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=1000,
    tau=0.005,  # the update rate of the target network, was 0.005
    reward_params=dict(
        w_dist_change=-0.5,
        w_kill_prey=1.,
        w_kill_enemy=3.,
        w_kill_bonus=1.3,
        standing_still_penalty=-0.7,
        gamma_for_bonus_count=0.5,
        n_nearest_targets=2,
    ),
    seed=1234
)


class Agent:
    def __init__(self):
        self.model = DQN(global_config, train_config)
        self.model.load(__file__[:-8] + '/model_weights.pt')

    def get_actions(self, state, info):
        processed_state = preprocess(state, info)
        actions = self.model.get_actions(processed_state)
        return actions

    def reset(self, state, info):
        pass

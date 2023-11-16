import copy
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class DQN(nn.Module):
    def __init__(self, n_masks, n_actions, n_predators, map_size, device, config):
        self.n_masks = n_masks
        self.n_actions = n_actions
        self.n_predators = n_predators
        self.map_size = map_size
        self.device = device
        self.cfg = config
        super().__init__()

        self.dqn = nn.Sequential(
            nn.Conv2d(self.n_masks, 4, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(6, 1, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(40 * 40 // 2**6, self.n_actions),
        )

        self.steps_done = 0
        self.target_dqn = copy.deepcopy(self.dqn)
        self.buffer = deque(maxlen=self.cfg.buffer_size)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.dqn.parameters(), lr=self.cfg.learning_rate, amsgrad=True)
        self.q_values = [None] * n_predators # used for display in gif

    def get_actions(self, processed_state, random=False):
        if random:
            return torch.randint(0, self.n_actions, (self.n_predators,))
        else:
            # expects unbatched input: [self.n_predators, self.n_masks, self.map_size, self.map_size]
            with torch.no_grad():
                # [self.n_predators, self.n_actions]
                self.q_values = self.dqn(torch.FloatTensor(processed_state))
                return self.q_values.argmax(dim=1)

    def consume_transition(self, *args):
        self.buffer.append(Transition(*args))

    def sample_batch(self):
        processed_state = torch.empty(self.cfg.batch_size, self.n_predators, self.n_masks,
                                      self.map_size, self.map_size, device=self.device, dtype=torch.float32)
        actions = torch.empty(self.cfg.batch_size, self.n_predators, device=self.device, dtype=torch.int32)
        next_processed_state = torch.empty(self.cfg.batch_size, self.n_predators, self.n_masks,
                                           self.map_size, self.map_size, device=self.device, dtype=torch.float32)
        reward = torch.empty(self.cfg.batch_size, self.n_predators, device=self.device, dtype=torch.float32)
        done = torch.empty(self.cfg.batch_size, device=self.device, dtype=torch.bool)

        for i, t in enumerate(random.sample(self.buffer, self.cfg.batch_size)):            
            processed_state[i] = torch.from_numpy(t.state)
            actions[i] = torch.tensor(t.action, device=self.device, dtype=torch.int32)
            next_processed_state[i] = torch.from_numpy(t.next_state)
            reward[i] = torch.tensor(t.reward, device=self.device, dtype=torch.float32)
            done[i] = t.done

        return processed_state, actions, next_processed_state, reward, done

    def update_policy_network(self):
        # processed_state: [BATCH_SIZE, self.n_predators, self.n_masks, self.map_size, self.map_size]
        processed_state, actions, next_processed_state, reward, done = self.sample_batch()

        q_values = []
        for i in range(self.cfg.batch_size):
            # [self.n_predators, self.n_actions] for all actions
            preds = self.dqn(processed_state[i])
            q_values.append(preds[torch.arange(self.n_predators), actions[i]])
        # [BATCH_SIZE, self.n_predators] for taken actions
        q_values = torch.stack(q_values)

        target_q_values = []
        with torch.no_grad():
            for i in range(self.cfg.batch_size):
                # [self.n_predators, self.n_actions] for all actions
                preds = self.target_dqn(next_processed_state[i])
                future_q_values = preds.max(dim=1).values if not done[i] else torch.zeros(self.n_predators)  # [self.n_predators]
                target_q_values.append(reward[i] + self.cfg.gamma * future_q_values)
        # [BATCH_SIZE, self.n_predators]
        target_q_values = torch.stack(target_q_values)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.dqn.parameters(), 100)
        self.optimizer.step()

        return reward.mean(), loss.item()

    def soft_update_target_network(self):
        target_net_state_dict = self.target_dqn.state_dict()
        policy_net_state_dict = self.dqn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.cfg.tau + target_net_state_dict[key]*(1-self.cfg.tau)
        self.target_dqn.load_state_dict(target_net_state_dict)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

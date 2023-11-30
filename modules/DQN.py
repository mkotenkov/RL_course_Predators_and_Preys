import copy
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class InnerModel(nn.Module):
    def __init__(self, global_config, n_neurons_to_process_bonus=32):
        super().__init__()        
        self.n_masks = global_config.n_masks
        self.n_actions = global_config.n_actions
        self.n_predators = global_config.n_predators  
        self.map_size = global_config.map_size     

        self.backbone = nn.Sequential(
            nn.Conv2d(self.n_masks, 4, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(6, 6, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(6, 1, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1)
        )

        self.backbone_3x3 = nn.Sequential(
            nn.Conv2d(self.n_masks, 32, kernel_size=3, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )

        self.backbone_5x5 = nn.Sequential(
            nn.Conv2d(self.n_masks, 32, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )

        self.backbone_9x9 = nn.Sequential(
            nn.Conv2d(self.n_masks, 16, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )

        self.backbone_17x17 = nn.Sequential(
            nn.Conv2d(self.n_masks, 4, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )

        self.bonus_count_processor = nn.Sequential(
            nn.Linear(1, n_neurons_to_process_bonus),
            nn.LeakyReLU()            
        )

        self.head = nn.Sequential(            
            nn.Linear(40 * 40 // 2**6 + n_neurons_to_process_bonus + 128, self.n_actions),
        )

    def forward(self, processed_state):        
        state, bonus_counts = processed_state
        
        state_3x3 = self.get_cut_from_processed_state(state, cut_size=3)
        state_5x5 = self.get_cut_from_processed_state(state, cut_size=5) 
        state_9x9 = self.get_cut_from_processed_state(state, cut_size=9) 
        state_17x17 = self.get_cut_from_processed_state(state, cut_size=17)               
        
        features_main = self.backbone(state)
        features_3x3 = self.backbone_3x3(state_3x3)
        features_5x5 = self.backbone_5x5(state_5x5)
        features_9x9 = self.backbone_9x9(state_9x9)
        features_17x17 = self.backbone_17x17(state_17x17)

        bonus_features = self.bonus_count_processor(bonus_counts.resize(self.n_predators, 1))        

        all_features = torch.cat((features_main, features_3x3, features_5x5, features_9x9, features_17x17, bonus_features), dim=1)                
        x = self.head(all_features)        
        return x
    
    def get_cut_from_processed_state(self, processed_state, cut_size):
        assert cut_size % 2 == 1
        assert processed_state.shape == (self.n_predators, self.n_masks, self.map_size, self.map_size)

        start = self.map_size // 2 - cut_size // 2        
        end = start + cut_size

        return processed_state[..., start:end, start:end]


class DQN(nn.Module):
    def __init__(self, global_config, train_config):
        self.global_config = global_config
        self.train_config = train_config

        self.n_masks = global_config.n_masks
        self.n_actions = global_config.n_actions
        self.n_predators = global_config.n_predators
        self.map_size = global_config.map_size
        self.device = global_config.device

        self.learning_rate = train_config.learning_rate
        self.buffer_size = train_config.buffer_size
        self.gamma = train_config.gamma
        self.batch_size = train_config.batch_size
        self.tau = train_config.tau
        self.steps = train_config.steps

        super().__init__()

        self.dqn = InnerModel(global_config)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self.buffer = deque(maxlen=self.buffer_size)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.dqn.parameters(), lr=self.learning_rate, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.steps)
        self.q_values = None  # used for display in gif

    def get_actions(self, processed_state, random=False):
        if random:
            return torch.randint(0, self.n_actions, (self.n_predators,))
        else:
            # expects unbatched input: [self.n_predators, self.n_masks, self.map_size, self.map_size]
            with torch.no_grad():
                # [self.n_predators, self.n_actions]
                self.q_values = self.dqn((torch.FloatTensor(processed_state[0]), torch.FloatTensor(processed_state[1])))
                return self.q_values.argmax(dim=1)

    def consume_transition(self, *args):
        self.buffer.append(Transition(*args))

    def sample_batch(self):
        processed_state = (torch.empty(self.batch_size, self.n_predators, self.n_masks,
                                       self.map_size, self.map_size, device=self.device, dtype=torch.float32),
                           torch.empty(self.batch_size, self.n_predators, device=self.device, dtype=torch.float32))
        actions = torch.empty(self.batch_size, self.n_predators, device=self.device, dtype=torch.int32)
        next_processed_state = (torch.empty(self.batch_size, self.n_predators, self.n_masks,
                                            self.map_size, self.map_size, device=self.device, dtype=torch.float32),
                                torch.empty(self.batch_size, self.n_predators, device=self.device, dtype=torch.float32))
        reward = torch.empty(self.batch_size, self.n_predators, device=self.device, dtype=torch.float32)
        done = torch.empty(self.batch_size, device=self.device, dtype=torch.bool)

        for i, t in enumerate(random.sample(self.buffer, self.batch_size)):
            processed_state[0][i] = torch.from_numpy(t.state[0])
            processed_state[1][i] = torch.from_numpy(t.state[1])
            actions[i] = torch.tensor(t.action, device=self.device, dtype=torch.int32)
            next_processed_state[0][i] = torch.from_numpy(t.next_state[0])
            next_processed_state[1][i] = torch.from_numpy(t.next_state[1])
            reward[i] = torch.tensor(t.reward, device=self.device, dtype=torch.float32)
            done[i] = t.done

        return processed_state, actions, next_processed_state, reward, done

    def update_policy_network(self):
        # processed_state: (
        # [BATCH_SIZE, self.n_predators, self.n_masks, self.map_size, self.map_size],  - masks
        # [BATCH_SIZE, self.n_predators]  - bonus info
        # )
        processed_state, actions, next_processed_state, reward, done = self.sample_batch()

        q_values = []
        for i in range(self.batch_size):
            # [self.n_predators, self.n_actions] for all actions
            preds = self.dqn((processed_state[0][i], processed_state[1][i]))
            q_values.append(preds[torch.arange(self.n_predators), actions[i]])
        # [BATCH_SIZE, self.n_predators] for taken actions
        q_values = torch.stack(q_values)

        target_q_values = []
        with torch.no_grad():
            for i in range(self.batch_size):
                # [self.n_predators, self.n_actions] for all actions
                preds = self.target_dqn((next_processed_state[0][i], next_processed_state[1][i]))
                future_q_values = preds.max(dim=1).values if not done[i] else torch.zeros(self.n_predators)
                target_q_values.append(reward[i] + self.gamma * future_q_values)
        # [BATCH_SIZE, self.n_predators]
        target_q_values = torch.stack(target_q_values)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.dqn.parameters(), 50)
        self.optimizer.step()

        return loss.item()

    def soft_update_target_network(self):
        target_net_state_dict = self.target_dqn.state_dict()
        policy_net_state_dict = self.dqn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_dqn.load_state_dict(target_net_state_dict)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

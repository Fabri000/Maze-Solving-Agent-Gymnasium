import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from collections import namedtuple

from lib.replay_memory import  ReplayMemory

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self,n_observations:int,n_actions:int,hidden_dim:int=64):
        super(DQN,self).__init__()
        self.model =nn.Sequential(
            nn.Linear(n_observations,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,n_actions)
        )
    
    def forward(self,x):
        return self.model(x)
    
class DQNAgent():
    def __init__(self,
        env,
        learning_rate:float,
        starting_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_factor: float,
        batch_size:int,
        memory_size:int,
        target_update_frequency:int,
        device,
        ):
        
        self.env = env
        self.device = device

        #parameters
        self.learning_rate = learning_rate
        self.starting_epsilon = starting_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        n_actions = env.action_space.n
        observation, _ = env.reset()
        n_observations = len(np.concatenate([observation[k] for k in observation]))

        self.source_net = DQN(n_observations,n_actions).to(device)
        self.target_net = DQN(n_observations,n_actions).to(device)

        self.memory = ReplayMemory(memory_size)

        self.optimizer = optim.AdamW(self.source_net.parameters(),learning_rate)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=100,eta_min=1e-5)
        self.steps_done = 0
    
    def memorize(self,*args):
        self.memory.push(*args)
    
    def select_action(self, state):
        sample = random.random()
        epsilon_threshold = self.final_epsilon + (self.starting_epsilon - self.final_epsilon) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample < epsilon_threshold:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.source_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        device = state_batch.device
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch.reward).to(device)

        state_action_values = self.source_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute loss between our state action and expectations
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.source_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
    
    def scheduler_step(self):
        self.lr_scheduler.step()
    
    def has_to_update(self,episode:int):
        if episode % self.target_update_frequency:
            return True
        else:
            return False

    def update_target(self):
        self.target_net.load_state_dict(self.source_net.state_dict()) 
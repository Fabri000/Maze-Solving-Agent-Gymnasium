import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.modules import Conv2d, ReLU,  MaxPool2d

from collections import namedtuple

from lib.replay_memory import  ReplayMemory

Transition = namedtuple('Transition', ('state', 'action','reward', 'next_state'))

class DQN(nn.Module):
    WINDOW_SIZE=(15,15)

    def __init__(self,in_channels:int,n_observations:int,n_actions:int,h_channels:int,hidden_dim:int=1024):
        super(DQN,self).__init__()

        self.in_channels = in_channels

        self.conv =nn.Sequential(
            Conv2d(in_channels,h_channels,kernel_size=3,stride=1,padding=1),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(h_channels, h_channels*2,kernel_size=3,stride=1,padding=1),
            ReLU(),
            MaxPool2d(2,2),
        )

        input_dim = self.get_conv_size(DQN.WINDOW_SIZE)+n_observations

        self.fc =nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim//2,n_actions)
        )

        self.conv.train()
        self.fc.train()
    
    def forward(self,x):
        s,w= x
        fw = self.conv(w)
        fw = fw.view(fw.shape[0], -1)
        y = torch.cat((fw,s),dim=1)
        return self.fc(y)
    
    def get_conv_size(self, shape):
        out_conv = self.conv(torch.zeros(1,self.in_channels, shape[0], shape[1]))
        return int(np.prod(out_conv.size()))
    
class DDQNAgent():
    def __init__(self,
        env,
        learning_rate:float,
        starting_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_factor: float,
        eta:float,
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
        self.eta = eta
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        n_actions = env.action_space.n
        observation, _ = env.reset()
        n_observations = len(np.concatenate([observation[k] for k in observation if k != "window"]))

        self.source_net = DQN(4,n_observations,n_actions,16).to(device)
        self.target_net = DQN(4,n_observations,n_actions,16).to(device)

        self.memory = ReplayMemory(memory_size)

        self.optimizer = optim.AdamW(self.source_net.parameters(),learning_rate)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=100,eta_min=1e-6)
        self.steps_done = 0
    
    def memorize(self,*args):
        self.memory.push(*args)
    
    def get_action(self, state):
        sample = random.random()
        epsilon_threshold = self.final_epsilon + (self.starting_epsilon - self.final_epsilon) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample < epsilon_threshold:
            return torch.tensor(self.env.action_space.sample(), device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return F.softmax(self.source_net(state)).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = (torch.cat([s[0] for s in batch.next_state if s is not None]),torch.cat([s[1] for s in batch.next_state if s is not None]))

        state_batch = (torch.cat([s[0] for s in batch.state]),torch.cat([s[1] for s in batch.state],dim=0))
        device = state_batch[0].device
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch.reward).to(device)

        state_action_values = self.source_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        best_actions = self.source_net(non_final_next_states).max(1)[1].unsqueeze(1)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1,best_actions).squeeze(1).detach()

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute loss between our state action and expectations
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        ret = loss.item()

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.source_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return ret
    
    def scheduler_step(self):
        self.lr_scheduler.step()
    
    def has_to_update(self,episode:int):
        if episode % self.target_update_frequency == 0:
            return True
        else:
            return False

    def update_target(self):
        self.target_net.load_state_dict(self.source_net.state_dict()) 
    
    def update_steps_done(self):
        self.steps_done = self.steps_done // 4
    
    def update_hyperparameter(self,is_better:bool):
        if is_better:
            self.discount_factor = self.discount_factor + self.eta
        else:
            self.discount_factor = self.discount_factor - self.eta

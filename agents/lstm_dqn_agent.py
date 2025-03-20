import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from collections import namedtuple

from lib.replay_memory import  SequentialExperienceReplayMemory

Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state'))

class DQN(nn.Module):

    def __init__(self,input_size:int,n_actions:int,hidden_size:int,device):
        super(DQN,self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size,hidden_size,device=device)
        self.fc = nn.Linear(hidden_size,n_actions).to(device)
        self.hidden_state = None
        self.cell_state = None
        
    def forward(self, x):
        """
        x: Input sequence (batch_size, sequence_length, input_size)
        hidden_state, cell_state: Initial hidden and cell states
        """
        _, seq_len, _ = x.size()

        for i in range(seq_len):
            self.hidden_state, self.cell_state = self.lstm_cell(x[:, i, :], (self.hidden_state, self.cell_state))

        q_values = self.fc(self.hidden_state)

        return q_values
    
    def reset_hidden_state(self, batch_size):
        self.hidden_state = torch.zeros(batch_size,self.hidden_size).to(self.device)
        self.cell_state = torch.zeros(batch_size,self.hidden_size).to(self.device)
    
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
        
        obs,_ = env.reset()
        n_observation = len(np.concatenate([obs[k] for k in obs.keys()], axis=0))
        n_actions = env.action_space.n

        self.source_net = DQN(n_observation,n_actions,32,device).to(device)
        self.source_net.reset_hidden_state(1)
        self.target_net = DQN(n_observation,n_actions,32,device).to(device)
        self.target_net.reset_hidden_state(1)

        self.memory = SequentialExperienceReplayMemory(memory_size)

        self.optimizer = optim.AdamW(self.source_net.parameters(),learning_rate)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=30,eta_min=1e-6)
        self.steps_done = 0
    
    def memorize(self,*args):
        self.memory.add(*args)
    
    def get_action(self, state):
        sample = random.random()
        epsilon_threshold = self.final_epsilon + (self.starting_epsilon - self.final_epsilon) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if sample < epsilon_threshold:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                print(state.shape)
                return F.softmax(self.source_net(state.unsqueeze(0))).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        self.source_net.reset_hidden_state(self.batch_size)
        self.target_net.reset_hidden_state(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s[0] for s in batch.next_state if s is not None])

        state_batch = torch.cat([s for s in batch.state])
        device = state_batch[0].device
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch.reward).to(device)

        state_action_values = self.source_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

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
        self.steps_done = self.steps_done // 2
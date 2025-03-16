import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self,n_observations:int,n_actions:int, h_dim:int):

        super(PolicyNetwork,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,n_actions),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.net(x)

class RFAgent():
    def __init__(self,env,lr:float,gamma:float,device):
        self.lr = lr
        self.gamma = gamma
        self.env = env
        self.device = device

        n_actions = env.action_space.n
        observation, _ = env.reset()
        n_observations = len(np.concatenate([observation[k] for k in observation]))

        self.policy_net = PolicyNetwork(n_observations,n_actions,h_dim=128).to(device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(),lr)

    def get_probs(self,state):
        return self.policy_net(state)
    
    def select_action(self,state):
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()

        return action.item(),probs

    def get_returns(self,rs):
        R=0
        returns = []
        for r in rs[::-1]:
            R = r + self.gamma * R
            returns.append(R)
        
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns-returns.mean())/(returns.std()+1e-9)

        return returns
    
    def optimize_model(self, probabilities, returns):
        probs = torch.stack(probabilities, dim=0).squeeze(1)
        returns = returns.to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = -(probs * returns.unsqueeze(1)).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.modules import Conv2d, ReLU,  MaxPool2d

class PolicyNetwork(nn.Module):
    WINDOW_SIZE=(15,15)
    def __init__(self,in_channels:int,n_actions:int,h_channels:int,hidden_dim:int=128):


        super(PolicyNetwork,self).__init__()
        self.in_channels = in_channels

        self.conv =nn.Sequential(
            Conv2d(in_channels,h_channels,kernel_size=3,stride=1),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(h_channels, h_channels,kernel_size=3,stride=1),
            ReLU(),
        )

        input_dim = self.get_conv_size(PolicyNetwork.WINDOW_SIZE)+4

        self.fc =nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim,n_actions),
            nn.Softmax()
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

class RFAgent():
    def __init__(self,env,lr:float,gamma:float,device):
        self.lr = lr
        self.gamma = gamma
        self.env = env
        self.device = device

        n_actions = env.action_space.n
        

        self.policy_net = PolicyNetwork(4,n_actions,32).to(device)

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
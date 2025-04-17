import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import Conv2d,  MaxPool2d
from torch.optim import lr_scheduler
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    WINDOW_SIZE=(15,15)

    def __init__(self,in_channels:int,n_observations:int,n_actions:int,h_channels:int,hidden_dim:int=1024):
        super(PolicyNetwork,self).__init__()

        self.in_channels = in_channels

        self.conv =nn.Sequential(
            Conv2d(in_channels,h_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            MaxPool2d(2,2),
        )

        input_dim = self.get_conv_size(PolicyNetwork.WINDOW_SIZE)+n_observations

        self.fc =nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim //2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim //2,n_actions)
        )

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):  # Check if the layer is a Conv2d layer
                nn.init.xavier_uniform_(layer.weight)

    
    def forward(self,x):
        s,w= x
        fw = self.conv(w)
        fw = fw.view(fw.shape[0], -1)
        y = torch.cat((fw,s),dim=1)
        scores = self.fc(y)
        return scores
    
    def get_conv_size(self, shape):
        out_conv = self.conv(torch.zeros(1,self.in_channels, shape[0], shape[1]))
        return int(np.prod(out_conv.size()))

class RFAgent():
    def __init__(self,env,lr:float,gamma:float,eta:float,device):
        self.lr = lr
        self.gamma = gamma
        self.eta = eta

        self.env = env
        self.device = device

        n_actions = env.action_space.n

        observation, _ = env.reset()
        n_observations = len(np.concatenate([observation[k] for k in observation if k != "window"]))

        self.policy_net = PolicyNetwork(3,n_observations,n_actions,32).to(device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(),lr)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=100,eta_min=1e-5)

    def get_probs(self,state):
        return self.policy_net(state)
    
    def select_action(self, state):
        state = (state[0].to(self.device), state[1].to(self.device))

        # Get logits with gradient tracking
        logits = self.policy_net(state)

        probs = F.softmax(logits,dim=1)

        action = torch.multinomial(probs, num_samples=1)

        # Calculate log probability using the original logits
        log_prob =  torch.log(probs[0,action.item()] + 1e-10)

        return action.item(), log_prob, torch.log(probs + 1e-10)

    def get_returns(self,rs):
        R=0
        returns = []
        for r in rs[::-1]:
            R = r + self.gamma * R
            returns.append(R)
        
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32,device=self.device)
        returns = (returns-returns.mean())/(returns.std()+1e-6)

        return returns
    
    def optimize_model(self, action_probabilities, probabilities, returns):
        
        probs = torch.stack(action_probabilities).to(self.device)  # Changed from stack to cat
        returns = returns.to(self.device)

        probabilities = torch.stack(probabilities).to(self.device)

        # Baseline: media dei ritorni
        baseline = returns.mean()
        advantages = returns - baseline

        # Calcola la loss con la baseline
        loss = -(probs * advantages.detach()).sum()
        
        # Manually zero and update gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def update_hyperparameter(self,is_better:bool):
        if is_better:
            self.gamma = self.gamma + self.eta
        else:
            self.gamma = self.gamma - self.eta
    
    def scheduler_step(self):
        self.lr_scheduler.step()

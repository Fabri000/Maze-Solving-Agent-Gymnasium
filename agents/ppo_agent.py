import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# After training, plot entropy values
import matplotlib.pyplot as plt

class ActorCriticNet(nn.Module):
    WINDOW_SIZE = (15,15)
    def __init__(self,in_channels:int,n_observations:int,n_actions:int,h_channels:int,hidden_dim:int=1024):
        super(ActorCriticNet,self).__init__()

        self.in_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,h_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
        )

        input_dim = self.get_conv_size(ActorCriticNet.WINDOW_SIZE)+n_observations

        self.actor_head = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim //2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim //2,n_actions)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim //2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim //2,1)
        )

    def forward(self,x):
        s,w= x
        fw = self.conv(w)
        fw = fw.view(fw.shape[0], -1)
        y = torch.cat((fw,s),dim=1)

        logit = self.actor_head(y)
        value = self.critic_head(y)
        return logit, value
    
    def act(self,state):
        """
        Execute an action on the given state.
        Args:
            state (tuple): state of the environment
        Returns:
            (int): action to execute,
            (torch.Tensor): log probability of the actions on the state estimated by the agent network
            (torch.Tensor): value of the actions estimated
        """
        action_pred,value_pred = self.forward(state)
        action_prob =  F.softmax(action_pred, dim=-1)
        action = torch.multinomial(action_prob, num_samples=1)

        log_prob = torch.log(action_prob.gather(1, action).squeeze(1))
        
        return action,log_prob,value_pred
    
    def evaluate(self,state,action):
        """
        Evaluate the action executed on a given state.
        Args:
            state (tuple): state of the environment
            action (int): the action executed
        Returns:
            (torch.Tensor): probability of actions,
            (torch.Tensor): value of the actions estimated
            (torch.Tensor): entropy
        """
        action_pred,value_pred = self.forward(state)
        action_prob =  F.softmax(action_pred, dim=-1)

        log_action_prob = F.log_softmax(action_pred, dim=-1)
        log_prob = log_action_prob.gather(1, action).squeeze(1)

        entropy = -torch.sum(action_prob * torch.log(action_prob + 1e-8), dim=1)

        return log_prob, value_pred, entropy
    
    def get_conv_size(self ,shape):
        feature = self.conv(torch.zeros(1,self.in_channels, shape[0], shape[1]))
        return int(np.prod(feature.size()))

class PPOAgent:
    def __init__(self,
                actor_lr:float,
                critic_lr:float,
                gamma:float,
                batch_size:int,
                ppo_steps:int,
                env,
                device,
                channels:int=3):
        torch.autograd.set_detect_anomaly(True)
        self.env = env
        self.device = device

        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.ppo_steps = ppo_steps
        

        n_actions = self.env.action_space.n

        observation, _ = env.reset()
        n_observations = len(np.concatenate([observation[k] for k in observation if k != "window"]))

        self.agent = ActorCriticNet(channels,n_observations,n_actions,32).to(device)

        self.optimizer = optim.AdamW([
            {'params': self.agent.actor_head.parameters(),'lr':self.actor_lr},
            {'params': self.agent.critic_head.parameters(),'lr':self.critic_lr},
            {'params':self.agent.conv.parameters(),'lr':(self.actor_lr+self.critic_lr)/2}
        ])


    def init_episode(self):
        states = []
        actions = []
        actions_log_probability = []
        values = []
        rewards = []
        done = False
        episode_reward = 0
        return states, actions, actions_log_probability, values, rewards, done, episode_reward
    
    def do_episode(self,):
        states, actions, actions_log_probability, values, rewards, done, episode_reward = self.init_episode()
        
        observation,_ = self.env.reset()
        win=False

        while not done:
            state = (torch.tensor(np.concatenate([observation[k] for k in observation.keys() if k!='window'], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0), observation['window'].to(self.device).unsqueeze(0))
            states.append(state)
            action,log_prob_action, value_pred = self.agent.act(state)
            actions.append(action)
            actions_log_probability.append(log_prob_action)
            values.append(value_pred)
            observation, reward,truncated,terminated,_ = self.env.step(action.item())
            rewards.append(reward)
            episode_reward += reward
            done = terminated or truncated
            win = terminated
        
        positions,windows =  zip(*states)
        states = (torch.cat(positions).to(self.device),torch.cat(windows).to(self.device))
        actions = torch.cat(actions).to(self.device)
        actions_log_probability = torch.stack(actions_log_probability).reshape(-1, 1).to(self.device)
        values = torch.cat(values).squeeze(-1).to(self.device)
        returns = self.calculate_returns(rewards).to(self.device)
        advantages = self.calculate_advantages(returns, values)
        return episode_reward, states, actions, actions_log_probability, advantages, returns, win

    def calculate_returns(self,rewards):
        returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + cumulative_reward * self.gamma
            returns.insert(0, cumulative_reward)
        returns = torch.tensor(returns)
        # normalize the return
        returns = (returns - returns.mean()) / (returns.std())
        return returns

    def calculate_advantages(self,returns,values):
        advantages = returns -  values

        # Normalize the advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def calculate_surrogate_loss(self,action_log_prob_old,action_log_prob_new,advantages,epsilon:float=3e-1):
        advantages=advantages.detach()

        policy_ratio = (action_log_prob_new - action_log_prob_old).exp()

        s_l_1 = policy_ratio * advantages
        s_l_2 = torch.clamp(policy_ratio,min=1-epsilon,max=1+epsilon)*advantages
        return torch.min(s_l_1,s_l_2).mean(), (action_log_prob_new - action_log_prob_old).mean()
    
    def calculate_loss(self,surrogate_loss, entropy, returns, value_pred,entropy_coefficient:float=1e-2):
        entropy_bonus = entropy * entropy_coefficient
        policy_loss = - (surrogate_loss+entropy_bonus).mean()

        value_loss = F.mse_loss(returns.unsqueeze(1), value_pred)
        return policy_loss, value_loss,entropy.mean()


    def optimize_model(self, states, actions, action_log_prob, advantages, returns,entropy_coef):
        action_log_probability = action_log_prob.detach()
        actions = actions.detach()
        training_results_dataset = TensorDataset(
            states[0],
            states[1],
            actions,
            action_log_probability,
            advantages,
            returns)
        batch_dataset = DataLoader(training_results_dataset,self.batch_size,shuffle=False)

        for _ in range(self.ppo_steps):
            for batch_idx, (positions,windows,actions,actions_log_probability,advantages,returns) in enumerate(batch_dataset):
                states  = (positions,windows)
                action_log_probability_new,value_pred,entropy =self.agent.evaluate(states,actions)
                surrogate_loss,kl_div = self.calculate_surrogate_loss(actions_log_probability,action_log_probability_new,advantages)
                policy_loss, value_loss, mean_entropy = self.calculate_loss(surrogate_loss,entropy,returns,value_pred,entropy_coef)
                total_loss = policy_loss + 0.5 * value_loss

                print(
                    f"Policy loss = {policy_loss.item():.4f} | "
                    f"Value loss = {value_loss.item():.4f} | "
                    f"Entropy = {mean_entropy.item():.4f} | "
                    f"Total loss = {total_loss.item():.4f} |"
                    f"Entropy coef = {entropy_coef:.4f} | "
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
                self.optimizer.step()

    def evaluate(self):
        self.agent.eval()
        done = False
        episode_reward = 0
        obs,_ = self.env.reset()
        while not done:
            state = (torch.tensor(np.concatenate([obs[k] for k in obs.keys() if k!='window'], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0), obs['window'].to(self.device).unsqueeze(0))
            with torch.no_grad():
                action_pred, _ = self.agent(state)
                action_prob = F.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, truncated, terminated, _ = self.env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
        return episode_reward,terminated, truncated
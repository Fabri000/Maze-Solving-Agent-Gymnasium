
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm


class ValueBasedTrainer():

    def __init__(self,env,agent,logger,device):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.device = device

    def train(self,n_episodes:int):

        for _ in tqdm(range(n_episodes)):
            observation,_ = self.env.reset()
            probabilities = []
            rewards = []
            done = False

            while not done:
                state = (torch.tensor(np.concatenate([observation[k] for k in observation.keys() if k!='window'], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0), observation['window'].to(self.device).unsqueeze(0))

                action,probs = self.agent.select_action(state)
                
                observation, reward, truncated, terminated, _ = self.env.step(action)
                
                done = terminated or truncated

                probabilities.append(probs.squeeze(0))
                
                rewards.append(reward)

            returns = self.agent.get_returns(rewards)

            self.agent.optimize_model(probabilities,returns)
        
        self.env.close()
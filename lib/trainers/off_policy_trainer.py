from gymnasium_env.envs.maze_env import MazeEnv
from gymnasium_env.envs.variable_maze_env import VariableMazeEnv

from tqdm import tqdm

import torch
import numpy as np

class OffPolicyTrainer():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self,n_episodes:int):
        # reset the environment to get the first observation
        done = False

        for episode in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            cumulative = 0

            # play one episode
            while not done:
                action = self.agent.get_action(self.env, obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                cumulative += reward

                # update the agent
                self.agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            print(f'episode {episode} shape ({len(self.env.env.maze_map)},{len(self.env.env.maze_map[0])}) cumulative reward {cumulative}')
            self.agent.decay_epsilon()
    
    def test(self, num_mazes:int):
        win = 0

        for _ in range(num_mazes):
            if isinstance(self.env.env, MazeEnv):
                self.env.env.update_maze()
            elif isinstance(self.env.env, VariableMazeEnv):
                self.env.env.update_maze(False)

            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.agent.get_action(self.env, obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    win += 1
                    done = True
                else:
                    done = truncated
                obs = next_obs
        
            print(f'total reward {total_reward} shape {self.env.env.current_shape}')
        
        print(f'winrate {(win / num_mazes)*100} %')

class NeuralOffPolicyTrainer():
    def __init__(self,agent,env,device):
        self.agent = agent
        self.env = env
        self.device = device

    def train(self,n_episodes:int):
        for episode in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            
            state = torch.tensor(np.concatenate([obs[k] for k in obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
            # play one episode
            while not done:
                action = self.agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                
                next_state = torch.tensor(np.concatenate([next_obs[k] for k in next_obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
                

                self.agent.memorize(state,action,next_state,reward)
                
                done = terminated or truncated

                state = next_state

                self.agent.optimize_model()

            self.agent.scheduler_step()

            if self.agent.has_to_update(episode):
                self.agent.update_target()
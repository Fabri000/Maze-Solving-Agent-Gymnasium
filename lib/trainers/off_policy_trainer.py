from gymnasium_env.envs.maze_env import MazeEnv
from gymnasium_env.envs.variable_maze_env import VariableMazeEnv

from tqdm import tqdm

import torch
import numpy as np

class OffPolicyTrainer():
    def __init__(self, env, agent,logger):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.is_maze_variable = isinstance(self.env.env, VariableMazeEnv) 

    def train(self,n_episodes:int):
        # reset the environment to get the first observation
        done = False
        maze_size =  (0,0)
        for episode in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            cumulative = 0
            maze_size =  self.env.env.get_maze_shape()
            # play one episode
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)
                cumulative += reward

                # update the agent
                self.agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            if self.is_maze_variable:
                self.logger.info(f'episode {episode} cumulative reward {cumulative} on maze of shape {maze_size}')
                if self.env.env.get_maze_shape() == self.env.env.max_shape:
                    self.logger.info(f'episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'episode {episode} cumulative reward {cumulative}')
            

        self.logger.info(f'End training')
    
    def test(self, num_mazes:int):
        win = 0

        for _ in range(num_mazes):
            if isinstance(self.env.env, MazeEnv):
                self.env.env.update_maze()
            elif isinstance(self.env.env, VariableMazeEnv):
                self.env.env.update_maze(False)

            obs, _ = self.env.reset()
            done = False
            lost = False
            total_reward = 0
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    if self.is_maze_variable:
                        self.logger.info(f'Won on maze of shape {self.env.env.get_maze_shape()} with cumulative reward {total_reward}')    
                    else:
                        self.logger.info(f'Won with cumulative reward {total_reward}')
                    win += 1
                    done = True
                else:
                    done = lost= truncated
                obs = next_obs

            if lost:
                if self.is_maze_variable:
                        self.logger.info(f'Lost on maze of shape {self.env.env.get_maze_shape()} with cumulative reward {total_reward}')
                else:
                        self.logger.info(f'Lost with cumulative reward {total_reward}')

        self.logger.info(f'End test with winrate {(win / num_mazes)*100} %')

class NeuralOffPolicyTrainer():
    def __init__(self,agent,env,device,logger):
        self.agent = agent
        self.env = env
        self.device = device
        self.logger = logger
        self.is_maze_variable = isinstance(self.env.env, VariableMazeEnv) 

    def train(self,n_episodes:int):
        cum_rew = 0
        maze_size = (0,0)
        for episode in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            maze_size =  self.env.env.get_maze_shape()
            state = torch.tensor(np.concatenate([obs[k] for k in obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)

            while not done:
                
                action = self.agent.get_action(state)
                next_obs, reward, truncated, terminated, _ = self.env.step(action.item())
                cum_rew +=reward
                
                next_state = torch.tensor(np.concatenate([next_obs[k] for k in next_obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)

                self.agent.memorize(state,action,next_state,reward)

                if terminated and self.is_maze_variable:
                    self.agent.update_epsilon_decay(self.env.env.get_maze_shape(),n_episodes)

                done = terminated or truncated

                state = next_state

                self.agent.optimize_model()

            if self.agent.has_to_update(episode):
                self.agent.update_target()

            if self.is_maze_variable:
                self.logger.info(f'episode {episode} cumulative reward {cum_rew} on maze of shape {maze_size}')
                if self.env.env.get_maze_shape() == self.env.env.get_max_shape():
                    self.logger.info(f'episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'episode {episode} cumulative reward {cum_rew}')
                
            cum_rew = 0
            self.agent.scheduler_step()

        self.logger.info(f'End of training')


    def test(self,num_mazes:int):
        win = 0
        for _ in range(num_mazes):
            if isinstance(self.env.env, MazeEnv):
                self.env.env.update_maze()
            elif isinstance(self.env.env, VariableMazeEnv):
                self.env.env.update_maze(False)
                
            obs, _ = self.env.reset()
            done = False
            lost = False
            total_reward = 0
            while not done:
                state = torch.tensor(np.concatenate([obs[k] for k in obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.agent.get_action(state)
                next_obs, reward, truncated, terminated, _ = self.env.step(action.item())
                total_reward += reward
                if terminated:
                    if self.is_maze_variable:
                        self.logger.info(f'won on maze of shape {self.env.env.get_maze_shape()} with total reward {total_reward}')
                    else:
                        self.logger.info(f'won with total reward {total_reward}')
                    win += 1
                    done = True
                else:
                    done = lost= truncated
                obs = next_obs

            if lost:
                self.logger.info(f'Not won with total reward {total_reward}')

        self.logger.info(f'End testing with total winrate {(win / num_mazes)*100}')
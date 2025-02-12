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
                self.logger.info(f'Episode {episode}: cumulative reward {cumulative} | maze of shape {maze_size}')
                if self.env.env.get_maze_shape() == self.env.env.max_shape:
                    self.logger.info(f'Episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {cumulative}')

        self.logger.info(f'End training')
    
    def test(self, num_mazes:int):
        win = 0
        self.logger.info(f'Start testing')
        for _ in range(num_mazes):
            if isinstance(self.env.env, MazeEnv):
                self.env.env.update_maze()
            elif isinstance(self.env.env, VariableMazeEnv):
                self.env.env.update_maze(False)

            obs, _ = self.env.reset()
            done = False

            maze_size =  self.env.env.get_maze_shape()
            lost = False
            total_reward = 0
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    
                    win += 1
                    done = True
                else:
                    done = lost= truncated
                obs = next_obs
            
            result = "Lost" if lost else "Win"
            if self.is_maze_variable:
                        self.logger.info(f'{result} | maze shape {maze_size} | cumulative reward {total_reward}')    
            else:
                        self.logger.info(f'{result} | cumulative reward {total_reward}')

        self.logger.info(f'End test | Win Rate {(win / num_mazes)*100} %')

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
            total_loss = 0
            num_step = 0
            win = False

            while not done:
                num_step+=1
                action = self.agent.get_action(state)
                next_obs, reward, truncated, terminated, _ = self.env.step(action.item())
                cum_rew +=reward
                
                next_state = torch.tensor(np.concatenate([next_obs[k] for k in next_obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)

                self.agent.memorize(state,action,next_state,reward)

                done = terminated or truncated
                win = terminated

                if done and self.is_maze_variable:
                    self.agent.update_epsilon_decay(self.env.env.get_maze_shape(),n_episodes)

                state = next_state

                loss = self.agent.optimize_model()

                if loss:
                    total_loss +=loss

            result = "Win" if win else "Lost"
            if self.is_maze_variable:
                self.logger.info(f'Episode {episode}: cumulative reward {cum_rew} | {result} | maze of shape {maze_size} | average loss {total_loss / num_step}')
                if self.env.env.get_maze_shape() == self.env.env.get_max_shape():
                    self.logger.info(f'Episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {cum_rew} | {result} | average loss {total_loss / num_step}')
                
            cum_rew = 0
            self.agent.scheduler_step()

            if self.agent.has_to_update(episode):
                self.agent.update_target()

        self.logger.info(f'End of training')


    def test(self,num_mazes:int):
        self.logger.info(f'Start of Testing')

        win = 0
        for _ in range(num_mazes):
            if isinstance(self.env.env, MazeEnv):
                self.env.env.update_maze()
            elif isinstance(self.env.env, VariableMazeEnv):
                self.env.env.update_maze(False)
                
            obs, _ = self.env.reset()
            done = False

            maze_size =  self.env.env.get_maze_shape()
            lost = False
            total_reward = 0
            while not done:
                state = torch.tensor(np.concatenate([obs[k] for k in obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.agent.get_action(state)
                next_obs, reward, truncated, terminated, _ = self.env.step(action.item())
                total_reward += reward
                if terminated:
                    win += 1
                    done = True
                else:
                    done = lost= truncated
                obs = next_obs

            result = "Lost" if lost else "Win"
            if self.is_maze_variable:
                        self.logger.info(f'{result} | maze shape {maze_size} | total reward {total_reward}')
            else:
                        self.logger.info(f'{result} | total reward {total_reward}')

        self.logger.info(f'End testing | total Win Rate {(win / num_mazes)*100}')
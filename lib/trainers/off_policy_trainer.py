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
            win = False
            # play one episode
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)
                cumulative += reward

                # update the agent
                self.agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                
                win = terminated

                obs = next_obs
            
            if win:
                    self.env.env.update_maze()

            win_status = "Win" if win else "Lost"
            if self.is_maze_variable:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cumulative,2)} | maze of shape {maze_size} | {win_status}')
                if self.env.env.get_maze_shape() == self.env.env.max_shape:
                    self.logger.info(f'Episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cumulative,2)} | {win_status}')

        self.logger.info(f'End training')
    
    def test(self, num_mazes:int):
        win_count = 0
        self.logger.info(f'Start testing')
        for _ in range(num_mazes):
            self.env.env.update_visited_maze()

            obs, _ = self.env.reset()
            done = False

            maze_size =  self.env.env.get_maze_shape()
            win = False
            total_reward = 0
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    win_count += 1
                    done = True
                    win = True
                else:
                    done = truncated
                obs = next_obs
            
            result = "Win" if win else "Lost"
            if self.is_maze_variable:
                        self.logger.info(f'{result} | maze shape {maze_size} | cumulative reward {round(total_reward,2)}')
            else:
                        self.logger.info(f'{result} | cumulative reward {round(total_reward,2)}')

        self.logger.info(f'End test | Win Rate {round(win_count / num_mazes,4) *100} %')
    
    def train_learned_maze(self, n_mazes:int):
        self.logger.debug("Training on already learned maze")

        for _ in range(n_mazes):
            self.env.env.update_visited_maze(remove = False)
            
            done = False
            maze_size =  (0,0)
            episode_count = 0
            win = False
            while not win:
                episode_count += 1
                obs, _ = self.env.reset()
                done = False
                cumulative = 0
                maze_size =  self.env.env.get_maze_shape()
                
                # play one episode
                while not done:
                    action = self.agent.get_action(obs)
                    next_obs, reward, truncated, terminated, _ = self.env.step(action)
                    cumulative += reward

                    self.agent.update(obs, action, reward, terminated, next_obs)

                    done = terminated or truncated
                    
                    win = terminated

                    obs = next_obs

                self.logger.info(f'Episode to learn to solve the maze {episode_count} | maze of shape {maze_size}')
        self.logger.debug("End learning on already learned mazes")
        

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
                    self.agent.update_epsilon_decay(self.env.env.get_maze_shape())

                state = next_state

                loss = self.agent.optimize_model()

                if loss:
                    total_loss +=loss
            
            if win:
                self.env.env.update_maze()

            result = "Win" if win else "Lost"
            if self.is_maze_variable:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cum_rew,2)} | {result} | maze of shape {maze_size} | average loss {round(total_loss / num_step,4)}')
                if self.env.env.get_maze_shape() == self.env.env.get_max_shape():
                    self.logger.info(f'Episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cum_rew,2)} | {result} | average loss {round(total_loss / num_step,4)}')
                
            cum_rew = 0
            self.agent.scheduler_step()

            if self.agent.has_to_update(episode):
                self.agent.update_target()

        self.logger.info(f'End of training')


    def test(self,num_mazes:int):
        self.logger.info(f'Start of Testing')

        win = 0
        for i in range(num_mazes):
            print(i)
            self.env.env.update_visited_maze()
                
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
                self.logger.info(f'{result} | maze shape {maze_size} | total reward {round(total_reward,4)}')
            else:
                self.logger.info(f'{result} | total reward {round(total_reward,4)}')

        self.logger.info(f'End testing | total Win Rate {round(win / num_mazes,4)*100}')


    def train_learned_maze(self, n_mazes:int):
        self.logger.debug("Training on already learned maze")
        
        for _ in range(n_mazes):
            self.env.env.update_visited_maze(remove = False)

            done = False
            maze_size =  (0,0)
            episode_count = 0
            win = False

            while not win:
                episode_count += 1
                obs, _ = self.env.reset()
                done = False
                maze_size =  self.env.env.get_maze_shape()
                
                state = torch.tensor(np.concatenate([obs[k] for k in obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
                # play one episode
                while True:
                    action = self.agent.get_action(state)
                    next_obs, reward, truncated, terminated, _ = self.env.step(action.item())
                    
                    next_state = torch.tensor(np.concatenate([next_obs[k] for k in next_obs], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)

                    self.agent.memorize(state,action,next_state,reward)

                    done = terminated or truncated
                    

                    if done and self.is_maze_variable:
                        self.agent.update_epsilon_decay(self.env.env.get_maze_shape())

                    if done:
                        win = terminated
                        break

                    state = next_state
                    self.agent.optimize_model()

            self.logger.info(f'Episode to learn to solve the maze {episode_count} | maze of shape {maze_size}')
        self.logger.debug("End learning on already learned mazes")
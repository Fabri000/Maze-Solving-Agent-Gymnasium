import random
import torch
import numpy as np
from tqdm import tqdm
from agents.ppo_agent import PPOAgent
from gymnasium_env.envs.simple_variable_maze_env import SimpleVariableMazeEnv
from gymnasium_env.envs.toroidal_maze_env import ToroidalMazeEnv
from gymnasium_env.envs.toroidal_variable_maze_env import ToroidalVariableMazeEnv
from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation

# After training, plot entropy values
import matplotlib.pyplot as plt


class Buffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.actions_log_probability = []
        self.advantages = []
        self.returns = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.actions_log_probability[:]
        del self.advantages[:]
        del self.returns[:]
    
    def add(self, states, actions, actions_log_probability, advantages, returns):
        self.states.append(states)
        self.actions.append(actions)
        self.actions_log_probability.append(actions_log_probability)
        self.advantages.append(advantages)
        self.returns.append(returns)
    
    def get(self):
        positions, windows = zip(*self.states)
        states = (torch.cat(positions),torch.cat(windows))
        return states,torch.cat(self.actions),torch.cat(self.actions_log_probability),torch.cat(self.advantages), torch.cat(self.returns)
    
    def __len__(self):
        len =  0
        for s in self.states:
            len += s[0].shape[0]
        return len

     
class PPOTrainer():
    ALGOS = ["r-prim","prim&kill","dfs"]
    def __init__(self,env,agent,logger,device):
        self.env = env
        self.agent:PPOAgent = agent
        self.logger = logger
        self.device = device

        self.buffer = Buffer()

        self.is_maze_variable = isinstance(self.env.env, SimpleVariableMazeEnv) or isinstance(self.env.env, ToroidalVariableMazeEnv)

    def train(self,n_episodes:int,update_freq:int):
        episode_reward = 0
        count_episode = 0
        num_win = 0

        for episode in tqdm(range(n_episodes)):
            count_episode+=1
            episode_reward, states, actions, actions_log_probability, advantages, returns, win = self.agent.do_episode()
            self.buffer.add(states, actions, actions_log_probability, advantages, returns)

            if episode % update_freq == 0 and episode!=0:
                states, actions, actions_log_probability, advantages, returns = self.buffer.get()
                entropy_coef = 1e-2 - (1e-2 - 5e-4) * (episode / n_episodes)
                self.agent.optimize_model(states, actions, actions_log_probability, advantages, returns,entropy_coef)
                
                self.buffer.clear()

            result = "Win" if win else "Lost"
            if self.is_maze_variable:
                self.logger.info(f'Episode {episode}: cumulative reward {round(episode_reward,2)} | {result} | maze of shape {self.env.env.get_maze_shape()} | generated using {self.env.env.get_algorithm()} | gamma {self.agent.gamma}')
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {round(episode_reward,2)} | {result} | gamma {self.agent.gamma}')

            if win:

                num_win+=1
                c_e = None
                if isinstance(self.env.env, ToroidalVariableMazeEnv) or isinstance(self.env.env, ToroidalMazeEnv):
                    maze =  np.pad(self.env.env.maze_map, pad_width=1, mode='constant', constant_values=0).tolist()
                    c_e = ComplexityEvaluation(maze,(self.env.env._start_pos[0]+1,self.env.env._start_pos[1]+1),tuple(self.env.env._target_location + 1))
                else:
                    c_e = ComplexityEvaluation(self.env.env.maze_map,self.env.env._start_pos,tuple(self.env.env._target_location))
                self.logger.debug(f'Episode to learn how to reach the goal {count_episode} | maze of shape {self.env.env.maze_shape}| generated using {self.env.env.get_algorithm()} | maze difficulty {c_e.difficulty_of_maze()}')
                count_episode = 0
                self.change_algorithm(num_win)
                self.env.env.update_maze()
                if self.is_maze_variable:
                    maze_size = self.env.env.get_maze_shape()
                    if isinstance(self.env.env, ToroidalVariableMazeEnv) or isinstance(self.env.env, ToroidalMazeEnv):
                        maze =  np.pad(self.env.env.maze_map, pad_width=1, mode='constant', constant_values=0).tolist()
                        c_e = ComplexityEvaluation(maze,(self.env.env._start_pos[0]+1,self.env.env._start_pos[1]+1),tuple(self.env.env._target_location + 1))
                    else:
                        c_e = ComplexityEvaluation(self.env.env.maze_map,self.env.env._start_pos,tuple(self.env.env._target_location))
                    if self.env.env.get_maze_shape() >= self.env.env.get_max_shape():
                        self.logger.info(f'Episode {episode} hitted max shape of maze')
                        return
                    else:
                        self.logger.debug(f'Learning new maze| maze of shape {maze_size} | generated using {self.env.env.get_algorithm()} | maze difficulty {c_e.difficulty_of_maze()}')
        

        self.env.close()
        self.logger.info(f'End of training')

    def test(self,num_mazes:int,new:bool):
        self.logger.info(f'Start of Testing')
        win = 0
        for _ in range(num_mazes):
            if new:
                algo= random.choice(PPOTrainer.ALGOS)
                self.env.env.set_algorithm(algo)
                self.env.env.update_new_maze()
            else:
                self.env.env.update_visited_maze(remove=True)

            episode_reward,terminated, truncated  = self.agent.evaluate()
            if terminated:
                win+=1
            
            result = "Win" if terminated else "Lost"
            if self.is_maze_variable:
                self.logger.info(f'{result} | maze shape {self.env.maze_shape} | total reward {round(episode_reward,4)} | algorithm {self.env.env.ALGORITHM}')
            else:
                self.logger.info(f'{result} | total reward {round(episode_reward,4)} | algorithm {self.env.env.ALGORITHM}')

        self.logger.info(f'End testing | total Win Rate {round(win / num_mazes,4)*100}')    

    def change_algorithm(self, num_win:int ):
        if num_win == 10:
            self.env.env.set_algorithm(PPOTrainer.ALGOS[2])
        elif num_win == 5:
            self.env.env.set_algorithm(PPOTrainer.ALGOS[1])
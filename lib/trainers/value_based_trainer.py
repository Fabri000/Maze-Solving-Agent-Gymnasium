
import random
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from gymnasium_env.envs.simple_variable_maze_env import SimpleVariableMazeEnv
from gymnasium_env.envs.toroidal_maze_env import ToroidalMazeEnv
from gymnasium_env.envs.toroidal_variable_maze_env import ToroidalVariableMazeEnv
from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation


class ValueBasedTrainer():
    ALGOS = ["r-prim","prim&kill","dfs"]

    def __init__(self,env,agent,logger,device):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.device = device

        self.is_maze_variable = isinstance(self.env.env, SimpleVariableMazeEnv) or isinstance(self.env.env, ToroidalVariableMazeEnv)

    def train(self,n_episodes:int):
        num_win=0
        count_episode = 0
        
        for episode in tqdm(range(n_episodes)):
            observation,_ = self.env.reset()
            log_probabilities_action = []
            log_probabilities = []
            rewards = []
            maze_size =  self.env.env.get_maze_shape()
            done = False
            win = False
            count_episode +=1

            while not done:
                state = (torch.tensor(np.concatenate([observation[k] for k in observation.keys() if k!='window'], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0), observation['window'].to(self.device).unsqueeze(0))

                action, prob, probs = self.agent.select_action(state)
                
                observation, reward, truncated, terminated, _ = self.env.step(action)
                
                done = terminated or truncated
                win = terminated

                log_probabilities_action.append(prob.squeeze(0))
                log_probabilities.append(probs.squeeze(0))
                
                rewards.append(reward)

            returns = self.agent.get_returns(rewards)

            result = "Win" if win else "Lost"
            if self.is_maze_variable:
                self.logger.info(f'Episode {episode}: cumulative reward {round(sum(rewards),2)} | {result} | maze of shape {self.env.maze_shape}')
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {round(sum(rewards),2)} | {result} | gamma {self.agent.gamma}')

            if win:
                num_win+=1
                c_e = None
                if isinstance(self.env.env, ToroidalVariableMazeEnv) or isinstance(self.env.env, ToroidalMazeEnv):
                    maze =  np.pad(self.env.env.maze_map, pad_width=1, mode='constant', constant_values=0).tolist()
                    c_e = ComplexityEvaluation(maze,(self.env.env._start_pos[0]+1,self.env.env._start_pos[1]+1),tuple(self.env.env._target_location + 1))
                else:
                    c_e = ComplexityEvaluation(self.env.env.maze_map,self.env.env._start_pos,tuple(self.env.env._target_location))
                self.logger.debug(f'Episode to learn how to reach the goal {count_episode} | maze of shape {maze_size}| generated using {self.env.env.get_algorithm()} | maze difficulty {c_e.difficulty_of_maze()}')
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

            self.agent.optimize_model(log_probabilities_action,log_probabilities,returns)
            
            self.agent.scheduler_step()
        
        self.env.close()
        self.logger.info(f'End of training')


    def test(self,num_mazes:int,new:bool):
        self.logger.info(f'Start of Testing')
        win = 0
        for _ in range(num_mazes):
            if new:
                algo= random.choice(ValueBasedTrainer.ALGOS)
                self.env.env.set_algorithm(algo)
                self.env.env.update_new_maze()
            else:
                self.env.env.update_visited_maze(remove=True)
                
            obs, _ = self.env.reset()
            done = False

            maze_size =  self.env.env.get_maze_shape()
            lost = False
            total_reward = 0
            while not done:
                state = (torch.tensor(np.concatenate([obs[k] for k in obs.keys() if k!='window'], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0), obs['window'].to(self.device).unsqueeze(0))
                action,_, _ = self.agent.select_action(state)
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
                self.logger.info(f'{result} | maze shape {maze_size} | total reward {round(total_reward,4)} | algorithm {self.env.env.ALGORITHM}')
            else:
                self.logger.info(f'{result} | total reward {round(total_reward,4)} | algorithm {self.env.env.ALGORITHM}')

        self.logger.info(f'End testing | total Win Rate {round(win / num_mazes,4)*100}')
    
    def change_algorithm(self, num_win:int ):
        if num_win == 10:
            self.env.env.set_algorithm( ValueBasedTrainer.ALGOS[2])
        elif num_win == 5:
            self.env.env.set_algorithm(ValueBasedTrainer.ALGOS[1])
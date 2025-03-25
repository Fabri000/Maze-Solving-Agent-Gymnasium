import random
from tqdm import tqdm
import torch
import numpy as np

from gymnasium_env.envs.simple_variable_maze_env import SimpleVariableMazeEnv
from gymnasium_env.envs.toroidal_variable_maze_env import ToroidalVariableMazeEnv
from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation

class OffPolicyTrainer():
    ALGOS = ["r-prim","prim&kill","dfs"]

    def __init__(self, env, agent, logger):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.is_maze_variable = isinstance(self.env.env, SimpleVariableMazeEnv) or isinstance(self.env.env, ToroidalVariableMazeEnv)
        self.algo_id= 0

    def train(self,n_episodes:int):
        # reset the environment to get the first observation
        done = False
        maze_size =  (0,0)
        count__episode = 0
        num_win = 0
        
        for episode in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            cumulative = 0
            prev_cum_rew = 0
            maze_size =  self.env.env.get_maze_shape()
            win = False
            count__episode += 1

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

            win_status = "Win" if win else "Lost"
            if self.is_maze_variable:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cumulative,2)} | maze of shape {maze_size} | {win_status}')
                if self.env.env.get_maze_shape() == self.env.env.get_max_shape():
                    self.logger.info(f'Episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cumulative,2)} | {win_status}')
            
            if win:
                c_e = ComplexityEvaluation(self.env.env.maze_map,self.env.env._start_pos,tuple(self.env.env._target_location))
                self.logger.debug(f'Episode to learn how to reach the goal {count__episode} | maze of shape {maze_size}| generated using {self.env.env.get_algorithm()} | maze difficulty {c_e.difficulty_of_maze()}')
                count__episode = 0
                num_win += 1
                #self.change_algorithm(num_win)
                self.env.env.update_maze()
            
            increment = cumulative > prev_cum_rew
            self.agent.update_hyperparameter(increment)
            prev_cum_rew = cumulative
            cumulative = 0

        self.logger.info(f'End training')
    
    def test(self, num_mazes:int, new:bool):
        win_count = 0
        num = num_mazes
        for _ in range(num_mazes):
            if new:
                #self.env.env.set_algorithm(random.choice(OffPolicyTrainer.ALGOS))
                self.env.env.update_new_maze()
            else:
                self.env.env.update_visited_maze(remove=True)

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

        self.logger.info(f'End test | Win Rate {round(win_count / num,4) *100} %')
    
    def change_algorithm(self,num_win:int):
        algo_id = 0
        if num_win > 10:
            algo_id = 2
        elif num_win >= 5:
            algo_id = 1
        self.env.env.set_algorithm(OffPolicyTrainer.ALGOS[algo_id])
        

class NeuralOffPolicyTrainer():
    ALGOS = ["r-prim","prim&kill","dfs"]

    def __init__(self,agent,env,device,logger):
        self.agent = agent
        self.env = env
        self.device = device
        self.logger = logger
        self.is_maze_variable = isinstance(self.env.env, SimpleVariableMazeEnv) or isinstance(self.env.env, ToroidalVariableMazeEnv)

    def train(self,n_episodes:int):
        cum_rew = 0
        prev_cum_rew=0
        maze_size = None
        count__episode = 0
        num_win = 0

        for episode in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            maze_size =  self.env.env.get_maze_shape()
            state = (torch.tensor(np.concatenate([obs[k] for k in obs.keys() if k!='window'], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0), obs['window'].to(self.device).unsqueeze(0))
            total_loss = 0
            num_step = 0
            win = False
            count__episode  += 1

            while not done:
                num_step+=1
                action = self.agent.get_action(state)
                next_obs, reward, truncated, terminated, _ = self.env.step(action.item())
                cum_rew +=reward
                
                next_state = (torch.tensor(np.concatenate([next_obs[k] for k in next_obs if k!='window'], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0), obs['window'].to(self.device).unsqueeze(0))

                self.agent.memorize(state,action,reward,next_state)

                done = terminated or truncated
                win = terminated

                state = next_state

                loss = self.agent.optimize_model()

                if loss:
                    total_loss +=loss
           
            result = "Win" if win else "Lost"

            if self.is_maze_variable:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cum_rew,2)} | {result} | maze of shape {maze_size}')
                if self.env.env.get_maze_shape() == self.env.env.get_max_shape():
                    self.logger.info(f'Episode {episode} hitted max shape of maze')
                    return
            else:
                self.logger.info(f'Episode {episode}: cumulative reward {round(cum_rew,2)} | {result} | ')
            
            if win:
                num_win+=1
                c_e = ComplexityEvaluation(self.env.env.maze_map,self.env.env._start_pos,tuple(self.env.env._target_location))
                self.logger.debug(f'Episode to learn how to reach the goal {count__episode} | maze of shape {maze_size}| generated using {self.env.env.get_algorithm()} | maze difficulty {c_e.difficulty_of_maze()}')
                count__episode = 0
                #self.change_algorithm(num_win)
                self.env.env.update_maze()
                if self.is_maze_variable:
                    self.agent.update_steps_done()
                    c_e = ComplexityEvaluation(self.env.env.maze_map,self.env.env._start_pos,tuple(self.env.env._target_location))
                    maze_size = self.env.env.get_maze_shape()
                    self.logger.debug(f'Learning new maze| maze of shape {maze_size} | generated using {self.env.env.get_algorithm()} | maze difficulty {c_e.difficulty_of_maze()}')
            
            increment = cum_rew > prev_cum_rew
            self.agent.update_hyperparameter(increment)
            prev_cum_rew = cum_rew
            cum_rew = 0
            self.agent.scheduler_step()

            if self.agent.has_to_update(episode):
                self.agent.update_target()

        self.logger.info(f'End of training')


    def test(self,num_mazes:int,new:bool):
        self.logger.info(f'Start of Testing')
        win = 0
        for _ in range(num_mazes):
            if new:
                #self.env.env.set_algorithm(random.choice(NeuralOffPolicyTrainer.ALGOS))
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

    def change_algorithm(self,num_win:int):
        algo_id = 0
        if num_win > 10:
            algo_id = 2
        elif num_win >= 5:
            algo_id = 1
        self.env.env.set_algorithm(OffPolicyTrainer.ALGOS[algo_id])
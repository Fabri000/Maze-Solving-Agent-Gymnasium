import os
import sys
import gymnasium as gym
import torch
import torch_directml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from gymnasium_env.envs.simple_variable_maze_env import SimpleEnrichVariableMazeEnv

from agents.rf_agent import RFAgent
from lib.trainers.value_based_trainer import ValueBasedTrainer
from lib.logger_inizializer import init_logger


max_shape = (35,35)

n_episodes = 200
lr = 1e-3
gamma = 0.99

device = torch_directml.device()

env = SimpleEnrichVariableMazeEnv(max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


agent = RFAgent(
    env=env,
    lr=lr,
    gamma= gamma,
    device= device
)

log_dir = "logs/reinforce_agent"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | total episodes of training {n_episodes}")
logger.debug(f"Hyperparameter of training: learning rate {lr} | gamma {gamma}")

trainer = ValueBasedTrainer(env,agent,logger,device)

trainer.train(n_episodes)

logger.info("Checking if the agent remember how to solve maze already seen")
trainer.test(len(env.env.mazes),new = False)
logger.info(f'Start testing on new mazes')
trainer.test(75, new = True)
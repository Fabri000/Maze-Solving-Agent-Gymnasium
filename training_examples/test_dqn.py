import sys
import os

# Get the absolute path to the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to sys.path
sys.path.append(root_dir)

import torch
import torch_directml
import gymnasium as gym

from gymnasium_env.envs.simple_maze_env import SimpleEnrichMazeEnv
from agents.dqn_agent import DQNAgent
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from lib.logger_inizializer import init_logger

maze_shape = (61,61)
device = torch_directml.device()

n_episodes = 150
learning_rate=1e-3
starting_epsilon=0.95
final_epsilon=0.05
epsilon_decay= maze_shape[0]*maze_shape[1]*4
discount_factor=0.5
eta = 1e-2
batch_size=128

env = SimpleEnrichMazeEnv(maze_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQNAgent(env,learning_rate=learning_rate,starting_epsilon=starting_epsilon,final_epsilon=final_epsilon,epsilon_decay=epsilon_decay,discount_factor=discount_factor,eta=eta,batch_size=batch_size,memory_size=10000,target_update_frequency=2,device=device)

log_dir = "logs/dqn_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | Difficulty {env.env.get_maze_difficulty()} | total episodes of training {n_episodes}")
logger.debug(f"Hyperparameters: lr {learning_rate} | eps_init {starting_epsilon} | eps_end {final_epsilon} | eps_dec {epsilon_decay} | discount_fact {discount_factor} | batch size {batch_size}")

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)


env.env.set_algorithm("r-prim")
trainer.train(n_episodes)

logger.info("Checking if the agent remember how to solve maze already seen")
trainer.test(len(env.env.mazes),new = False)
logger.info(f'Start testing on new mazes')
env.env.set_algorithm("prim&kill")
trainer.test(50, new = True)
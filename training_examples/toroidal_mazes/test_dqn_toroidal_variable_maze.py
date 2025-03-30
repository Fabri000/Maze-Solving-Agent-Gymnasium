import gymnasium as gym
import torch
import sys
import os

# Get the absolute path to the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to sys.path
sys.path.append(root_dir)

from gymnasium_env.envs.toroidal_variable_maze_env import ToroidalEnrichVariableMazeEnv

from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from agents.dqn_agent import DQNAgent
from lib.logger_inizializer import init_logger

import torch_directml

maze_max_shape = (31,31)
n_episodes = 500
learning_rate=1e-3
starting_epsilon=0.95
final_epsilon=0.05
epsilon_decay= maze_max_shape[0]*maze_max_shape[1]*4
discount_factor=0.99
batch_size=128

device = torch_directml.device()

env = ToroidalEnrichVariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQNAgent(env,learning_rate=learning_rate,starting_epsilon=1,final_epsilon=final_epsilon,epsilon_decay=epsilon_decay,discount_factor=discount_factor,batch_size=batch_size,memory_size=10000,target_update_frequency=2,device=device)

logger = init_logger("Agent_log","logs/tor_variable_dqn_logs")
logger.info(f"Training starting on variable mazes with dimension variable between {(15,15)} and {maze_max_shape}")
logger.debug(f"Hyperparameters: lr {learning_rate} | eps_init {starting_epsilon} | eps_end {final_epsilon} | eps_dec {epsilon_decay} | discount_fact {discount_factor} | batch size {batch_size}")

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

logger.info("Checking if the agent remember how to solve maze already seen")
trainer.test(len(env.env.mazes),new = False)
logger.info(f'Start testing on new mazes')
trainer.test(50, new = True)
logger.info(f'Test on different type of algos')
for algo in ["r-prim","prim&kill","dfs"]:
    trainer.infer(15,algo)
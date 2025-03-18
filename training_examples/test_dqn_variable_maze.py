import sys
import os

# Get the absolute path to the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to sys.path
sys.path.append(root_dir)

import gymnasium as gym
import torch
import torch_directml

from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from gymnasium_env.envs.simple_variable_maze_env import SimpleEnrichVariableMazeEnv
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from agents.dqn_agent import DQNAgent
from lib.logger_inizializer import init_logger

import torch_directml


maze_max_shape = (31,31)
n_episodes = 500
learning_rate=1e-1
starting_epsilon=0.9
final_epsilon=0.05
epsilon_decay= n_episodes * 2
discount_factor=0.99
batch_size=128

device = torch_directml.device()

env = SimpleEnrichVariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQNAgent(env,learning_rate=learning_rate,starting_epsilon=starting_epsilon,final_epsilon=final_epsilon,epsilon_decay=epsilon_decay,discount_factor=discount_factor,batch_size=batch_size,memory_size=10000,target_update_frequency=3,device=device)

logger = init_logger("Agent_log","logs/variable_dqn_logs")
logger.info(f"Training starting on variable mazes with dimension variable between {(7,7)} and {maze_max_shape}")
logger.debug(f"Hyperparameters: lr {learning_rate} | eps_init {starting_epsilon} | eps_end {final_epsilon} | eps_dec {epsilon_decay} | discount_fact {discount_factor} | batch size {batch_size}")

c_e = ComplexityEvaluation(env.env.maze_map,env.env._start_pos,tuple(env.env._target_location))
logger.debug(f'Learning new maze| maze of shape {env.env.maze_shape} | maze difficulty {c_e.difficulty_of_maze()}')

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

trainer.train(n_episodes)
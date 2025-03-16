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

maze_shape = (17,17)
device = torch_directml.device()

model = torch.load(f"weights/CAE_{(15,15)}.pth").to(device)
encoder = model.encoder.to(device)

n_episodes = 250
learning_rate=1e-3
starting_epsilon=1
final_epsilon=0.05
epsilon_decay=1250
discount_factor=0.99
batch_size=128

env = SimpleEnrichMazeEnv(maze_shape,encoder)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQNAgent(env,learning_rate=1e-3,starting_epsilon=starting_epsilon,final_epsilon=final_epsilon,epsilon_decay=epsilon_decay,discount_factor=discount_factor,batch_size=batch_size,memory_size=10000,target_update_frequency=2,device=device)

log_dir = "logs/dqn_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | total episodes of training {n_episodes}")
logger.debug(f"Hyperparameters: lr {learning_rate} | eps_init {starting_epsilon} | eps_end {final_epsilon} | eps_dec {epsilon_decay} | discount_fact {discount_factor} | batch size {batch_size}")

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

trainer.train(n_episodes)
trainer.train_learned_maze(len(env.env.mazes))
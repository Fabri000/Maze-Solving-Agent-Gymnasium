import gymnasium as gym

from gymnasium_env.envs.variable_maze_env import VariableMazeEnv
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from agents.dqn_agent import DQNAgent
from lib.logger_inizializer import init_logger

import torch_directml


maze_max_shape = (31,31)
n_episodes = 500
learning_rate=1e-3
starting_epsilon=1
final_epsilon=0.05
epsilon_decay= n_episodes
discount_factor=0.99
batch_size=128

env = VariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

device = torch_directml.device()

agent = DQNAgent(env,learning_rate=1e-3,starting_epsilon=1,final_epsilon=final_epsilon,epsilon_decay=epsilon_decay,discount_factor=discount_factor,batch_size=batch_size,memory_size=10000,target_update_frequency=2,device=device)

logger = init_logger("Agent_log","logs/variable_dqn_logs")

logger.info(f"Training starting on variable mazes with dimension variable between {(7,7)} and {maze_max_shape}")
logger.debug(f"Hyperparameters: lr {learning_rate} | eps_init {starting_epsilon} | eps_end {final_epsilon} | eps_dec {epsilon_decay} | discount_fact {discount_factor} | batch size {batch_size}")

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

trainer.train(n_episodes)
trainer.test(50)
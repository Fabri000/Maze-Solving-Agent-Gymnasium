import os
import logging
from datetime import datetime
import gymnasium as gym

from gymnasium_env.envs.variable_maze_env import VariableMazeEnv
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from agents.dqn_agent import DQNAgent

import torch_directml


env = VariableMazeEnv(max_shape=(31,31))

n_episodes = 250

device = torch_directml.device()

agent = DQNAgent(env,learning_rate=1e-2,starting_epsilon=0.9,final_epsilon=0.05,epsilon_decay=1 / (n_episodes / 2),discount_factor=0.99,batch_size=128,memory_size=10000,target_update_frequency=50,device=device)

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


log_dir = "logs/variable_dqn_logs"
os.makedirs(log_dir, exist_ok=True)

file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(filename=f"{log_dir}/run_{file_name}.log",filemode="a",format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level= logging.DEBUG)
logger = logging.getLogger("Agent_log")

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

trainer.train(n_episodes)
trainer.test(50)
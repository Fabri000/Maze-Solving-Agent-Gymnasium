import os
import logging
from datetime import datetime
import gymnasium as gym

from gymnasium_env.envs.variable_maze_env import VariableMazeEnv
from lib.trainers.off_policy_trainer import OffPolicyTrainer
from lib.maze_generator import gen_maze
from agents.q_agent import QAgent


env = VariableMazeEnv(max_shape=(31,31))

n_episodes = 250

agent = QAgent(
    env=env,
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay= 1/ (n_episodes / 2),
    final_epsilon=0.1,
)

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

log_dir = "logs/variable_q_logs"
os.makedirs(log_dir, exist_ok=True)

file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(filename=f"{log_dir}/run_{file_name}.log",filemode="a",format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level= logging.DEBUG)
logger = logging.getLogger("Agent_log")

trainer = OffPolicyTrainer(env,agent,logger)

trainer.train(n_episodes)

trainer.test(50)

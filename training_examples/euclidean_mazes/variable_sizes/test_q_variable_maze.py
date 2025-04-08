import os
import sys
import gymnasium as gym

# Get the absolute path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Add the root directory to sys.path
sys.path.append(root_dir)

from gymnasium_env.envs.simple_variable_maze_env import SimpleVariableMazeEnv

from lib.trainers.off_policy_trainer import OffPolicyTrainer
from agents.dq_agent import DQAgent
from lib.logger_inizializer import init_logger

maze_max_shape=(153,153)

n_episodes = 300
lr = 1e-3
eps_init = 0.95
eps_end = 0.05
eps_dec = maze_max_shape[0]*maze_max_shape[1] // 2
gamma = 0.7
eta = 1e-2

env = SimpleVariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


agent = DQAgent(
    env=env,
    learning_rate=lr,
    initial_epsilon=eps_init,
    final_epsilon= eps_end,
    epsilon_decay= eps_dec,
    discount_factor=gamma,
    eta=eta
)

log_dir = "logs/variable_q_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Training starting on variable mazes with dimension variable between {SimpleVariableMazeEnv.START_SHAPE} and {maze_max_shape}")
logger.debug(f"Hyperparameter of training: learning rate {lr} | initial epsilon {eps_init} | final epsilon {eps_end} | decay {eps_dec}")

trainer = OffPolicyTrainer(env,agent,logger)

trainer.train(n_episodes)
logger.info("Checking if the agent remember how to solve maze already seen")
trainer.test(len(env.env.mazes),new = False)
logger.info(f'Start testing on new mazes')
trainer.test(15, new = True)
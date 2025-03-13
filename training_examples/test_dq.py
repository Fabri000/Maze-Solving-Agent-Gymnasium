import gymnasium as gym

from gymnasium_env.envs.simple_maze_env import SimpleMazeEnv
from lib.maze_generation import gen_maze
from agents.dq_agent import DQAgent
from lib.trainers.off_policy_trainer import OffPolicyTrainer
from lib.logger_inizializer import init_logger

maze_size = (27,27)

n_episodes = 150
lr = 1e-3
eps_init = 1
eps_end = 0.05
eps_dec = n_episodes

env = SimpleMazeEnv(maze_size)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQAgent(
    env=env,
    learning_rate=lr,
    initial_epsilon=eps_init,
    final_epsilon= eps_end,
    epsilon_decay= eps_dec,
)

log_dir = "logs/doub_q_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | total episodes of training {n_episodes}")
logger.debug(f"Hyperparameter of training: learning rate {lr} | initial epsilon {eps_init} | final epsilon {eps_end} | decay {eps_dec}")

trainer = OffPolicyTrainer(env,agent,logger)

trainer.train(n_episodes)
trainer.train_learned_maze(len(env.env.mazes))
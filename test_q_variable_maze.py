import gymnasium as gym

from gymnasium_env.envs.variable_maze_env import VariableMazeEnv
from lib.trainers.off_policy_trainer import OffPolicyTrainer
from agents.q_agent import QAgent
from lib.logger_inizializer import init_logger

initial_size = (7,7)
maze_max_shape=(41,41)
n_episodes = 125
lr = 1e-3
eps_init = 1
eps_end = 0.05
eps_dec = n_episodes


total_steps = initial_size[0]*initial_size[1]*n_episodes

env = VariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QAgent(
    env=env,
    learning_rate=lr,
    initial_epsilon=eps_init,
    final_epsilon=eps_end,
    epsilon_decay= eps_dec,
)

log_dir = "logs/variable_q_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Training starting on variable mazes with dimension variable between {(7,7)} and {maze_max_shape}")
logger.debug(f"Hyperparameter of training: learning rate {lr} | initial epsilon {eps_init} | final epsilon {eps_end} | decay {eps_dec}")

trainer = OffPolicyTrainer(env,agent,logger)

trainer.train(n_episodes)

trainer.test(50)

import gymnasium as gym

from gymnasium_env.envs.variable_maze_env import VariableMazeEnv
from lib.trainers.off_policy_trainer import OffPolicyTrainer
from agents.q_agent import QAgent
from lib.logger_inizializer import init_logger

initial_size = (7,7)
maze_max_shape=(41,41)
n_episodes = 125

total_steps = initial_size[0]*initial_size[1]*n_episodes

env = VariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QAgent(
    env=env,
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay= 0.25 * total_steps,
    final_epsilon=0.1,
)


log_dir = "logs/variable_q_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Training starting on variable mazes with dimension variable between {(7,7)} and {maze_max_shape}")

trainer = OffPolicyTrainer(env,agent,logger)

trainer.train(n_episodes)

trainer.test(50)

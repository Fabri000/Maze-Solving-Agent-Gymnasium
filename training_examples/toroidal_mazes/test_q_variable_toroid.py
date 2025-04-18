import gymnasium as gym

from gymnasium_env.envs.toroidal_variable_maze_env import ToroidalVariableMazeEnv
from lib.trainers.off_policy_trainer import OffPolicyTrainer
from agents.q_agent import QAgent
from lib.logger_inizializer import init_logger

maze_max_shape=(51,51)
n_episodes = 250
lr = 1e-2
eps_init = 1
eps_end = 0.05
eps_dec = n_episodes

env = ToroidalVariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QAgent(
    env=env,
    learning_rate=lr,
    initial_epsilon=eps_init,
    final_epsilon=eps_end,
    epsilon_decay= eps_dec,
)

log_dir = "logs/toroid_variable_q_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Training starting on variable mazes with dimension variable between {ToroidalVariableMazeEnv.START_SHAPE} and {maze_max_shape}")
logger.debug(f"Hyperparameter of training: learning rate {lr} | initial epsilon {eps_init} | final epsilon {eps_end} | decay {eps_dec}")

trainer = OffPolicyTrainer(env,agent,logger)

logger.info("Checking if the agent remember how to solve maze already seen")
trainer.test(len(env.env.mazes),new = False)
logger.info(f'Start testing on new mazes')
trainer.test(50, new = True)
logger.info(f'Test on different type of algos')
for algo in ["r-prim","prim&kill","dfs"]:
    trainer.infer(15,algo)
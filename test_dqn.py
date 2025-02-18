import torch_directml
import gymnasium as gym

from gymnasium_env.envs.maze_env import MazeEnv
from lib.maze_generation import gen_maze
from agents.dqn_agent import DQNAgent
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from lib.logger_inizializer import init_logger

start_pos, maze = gen_maze((23,23))
win_pos = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == 2][0]

device = torch_directml.device()

n_episodes = 125
learning_rate=1e-3
starting_epsilon=1
final_epsilon=0.05
epsilon_decay=125
discount_factor=0.99
batch_size=128

env = MazeEnv(maze,start_pos,win_pos)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQNAgent(env,learning_rate=1e-3,starting_epsilon=starting_epsilon,final_epsilon=final_epsilon,epsilon_decay=epsilon_decay,discount_factor=discount_factor,batch_size=batch_size,memory_size=10000,target_update_frequency=2,device=device)

log_dir = "logs/dqn_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | total episodes of training {n_episodes}")
logger.debug(f"Hyperparameters: lr {learning_rate} | eps_init {starting_epsilon} | eps_end {final_epsilon} | eps_dec {epsilon_decay} | discount_fact {discount_factor} | batch size {batch_size}")

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

trainer.train(n_episodes)
trainer.train_learned_maze(len(env.env.mazes))
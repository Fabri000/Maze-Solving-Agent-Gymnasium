import torch_directml
import math
import gymnasium as gym

from gymnasium_env.envs.maze_env import MazeEnv
from lib.maze_generator import gen_maze
from agents.dqn_agent import DQNAgent
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from lib.logger_inizializer import init_logger

start_pos, maze = gen_maze((15,15))
win_pos = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == 2][0]

'''start_pos = (3,1) # rows , columns
maze = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,1,0,1,0,1,1,1,0,1,0],
    [0,1,0,1,0,1,0,0,0,1,0,1,0],
    [0,1,1,1,1,1,1,1,0,1,1,2,0],
    [0,1,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,0,1,1,1,1,1,0,1,1,1,0],
    [0,1,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,1,1,0,1,0,1,1,1,0,1,0],
    [0,1,0,1,0,0,0,0,0,1,0,1,0],
    [0,1,0,1,1,1,0,1,0,1,1,1,0],
    [0,0,0,1,0,0,0,1,0,1,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
]
win_pos = (3,11)'''

device = torch_directml.device()

n_episodes = 125

env = MazeEnv(maze,start_pos,win_pos)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQNAgent(env,learning_rate=1e-3,starting_epsilon=1,final_epsilon=0.05,epsilon_decay=125,discount_factor=0.99,batch_size=2,memory_size=10000,target_update_frequency=25,device=device)

log_dir = "logs/dqn_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | total episodes of training {n_episodes}")

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

trainer.train(n_episodes)
trainer.test(50)
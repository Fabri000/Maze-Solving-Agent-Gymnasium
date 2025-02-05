import gymnasium as gym
from gymnasium_env.envs.maze_env import MazeEnv
from lib.maze_generator import gen_maze
from agents.dqn_agent import DQNAgent

import torch_directml 

start_pos, maze = gen_maze(21,21)
win_pos = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == 2][0]

"""start_pos = (1,1) # rows , columns
maze = [
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,1,1,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1,0,1,0],
    [0,1,0,2,0,1,1,1,1,1,0],
    [0,1,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0]
]
win_pos = (5,3)"""

device = torch_directml.device()

env = MazeEnv(maze,start_pos,win_pos)

agent = DQNAgent(env,learning_rate=1e-2,eps_start=0.9,eps_end=0.05,eps_decay=10000,gamma=0.99,tau=0.005,batch=128,device=device)

agent.train(200)
import gymnasium as gym
from gymnasium_env.envs.maze_env import MazeEnv
from gymnasium_env.envs.maze_generator import gen_maze
from agents.dqn_agent import DQNAgent

import torch_directml 

start_pos,lab = gen_maze(15,15)
win_pos = [(r, c) for r in range(len(lab)) for c in range(len(lab[0])) if lab[r][c] == 2][0]

device = torch_directml.device()

env = MazeEnv(lab,start_pos,win_pos)

agent = DQNAgent(env,learning_rate=1e-4,eps_start=0.9,eps_end=0.05,eps_decay=1000,gamma=0.99,tau=0.005,batch=128,device=device)

agent.train(1000)
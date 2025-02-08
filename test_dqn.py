from gymnasium_env.envs.maze_env import MazeEnv
from lib.maze_generator import gen_maze
from agents.dqn_agent import DQNAgent
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer

import torch_directml 

'''start_pos, maze = gen_maze((11,11))
win_pos = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == 2][0]'''

start_pos = (1,1) # rows , columns
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
win_pos = (5,3)

device = torch_directml.device()

env = MazeEnv(maze,start_pos,win_pos)

agent = DQNAgent(env,learning_rate=1e-2,starting_epsilon=0.9,final_epsilon=0.05,epsilon_decay=10000,discount_factor=0.99,batch_size=128,memory_size=10000,target_update_frequency=50,device=device)

trainer = NeuralOffPolicyTrainer(agent,env,device)

trainer.train(200)
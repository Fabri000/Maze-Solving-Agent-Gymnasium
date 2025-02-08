import os
import logging
from datetime import datetime
import gymnasium as gym

from gymnasium_env.envs.maze_env import MazeEnv
from lib.maze_generator import gen_maze
from agents.q_agent import QAgent
from lib.trainers.off_policy_trainer import OffPolicyTrainer

start_pos,maze = gen_maze((21,21))
win_pos = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == 2][-1]

'''start_pos = (1,1) # rows , columns
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
win_pos = (5,3)'''

env = MazeEnv(maze,start_pos,win_pos)

n_episodes = 100

agent = QAgent(
    env=env,
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay= 1 / (n_episodes/2),
    final_epsilon=0.1,
)

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

log_dir = "logs/q_logs"
os.makedirs(log_dir, exist_ok=True)

file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(filename=f"{log_dir}/run_{file_name}.log",filemode="a",format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level= logging.DEBUG)
logger = logging.getLogger("Agent_log")

trainer = OffPolicyTrainer(env,agent,logger)
trainer.train(n_episodes)

trainer.test(50)
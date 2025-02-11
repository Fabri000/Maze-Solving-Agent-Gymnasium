import gymnasium as gym

from gymnasium_env.envs.maze_env import MazeEnv
from lib.maze_generator import gen_maze
from agents.dq_agent import DQAgent
from lib.trainers.off_policy_trainer import OffPolicyTrainer
from lib.logger_inizializer import init_logger

maze_size = (15,15)
start_pos,maze = gen_maze(maze_size)
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


n_episodes = 150
total_steps = maze_size[0]*maze_size[1]*n_episodes

env = MazeEnv(maze,start_pos,win_pos)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DQAgent(
    env=env,
    learning_rate=1e-2,
    initial_epsilon=1.0,
    epsilon_decay=total_steps*0.25,
    final_epsilon=1e-1,
)

log_dir = "logs/doub_q_logs"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | total episodes of training {n_episodes}")

trainer = OffPolicyTrainer(env,agent,logger)

trainer.train(n_episodes)
trainer.test(30)
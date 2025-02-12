import gymnasium as gym

from gymnasium_env.envs.maze_env import MazeEnv
from lib.maze_generator import gen_maze
from agents.q_agent import QAgent
from lib.trainers.off_policy_trainer import OffPolicyTrainer
from lib.logger_inizializer import init_logger

maze_size = (51,51)
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

n_episodes = 100
lr = 1e-3
eps_init = 1
eps_end = 0.05
eps_dec = n_episodes

env = MazeEnv(maze,start_pos,win_pos)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


agent = QAgent(
    env=env,
    learning_rate=lr,
    initial_epsilon=eps_init,
    final_epsilon= eps_end,
    epsilon_decay= eps_dec,
)

log_dir = "logs/q_logs"
logger = init_logger("Agent_log",log_dir)

logger.debug(f"Hyperparameter of training: learning rate {lr} | initial epsilon {eps_init} | final epsilon {eps_end} | decay {eps_dec}")

trainer = OffPolicyTrainer(env,agent,logger)

trainer.train(n_episodes)

trainer.test(50)
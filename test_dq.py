import gymnasium as gym
from gymnasium_env.envs.maze_env import MazeEnv
from gymnasium_env.envs.maze_generator import gen_maze
from agents.dq_agent import DQAgent

from tqdm import tqdm

start_pos,maze = gen_maze(31,31)
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

learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = DQAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# reset the environment to get the first observation
done = False
observation, info = env.reset()

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    cumulative = 0

    # play one episode
    while not done:
        action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    print(f'episode {episode} cumulative reward {cumulative}')
    agent.decay_epsilon()
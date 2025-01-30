import gymnasium as gym
from gymnasium_env.envs.maze_env import MazeEnv
from gymnasium_env.envs.maze_generator import gen_maze
from agents.q_agent import QAgent

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm



start_pos,lab = gen_maze(15,15)
win_pos = [(r, c) for r in range(len(lab)) for c in range(len(lab[0])) if lab[r][c] == 2][0]
lab[win_pos[0]][win_pos[1]] = 2

env = MazeEnv(lab,start_pos,win_pos)

# hyperparameters
learning_rate = 0.01
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = QAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


# reset the environment to get the first observation
done = False
observation, info = env.reset()

# observation = (16, 9, False)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    cumulative = 0

    # play one episode
    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    print(f'episode {episode} cumulative reward {cumulative}')
    agent.decay_epsilon()
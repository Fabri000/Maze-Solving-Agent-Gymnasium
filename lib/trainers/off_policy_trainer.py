from gymnasium_env.envs.maze_env import MazeEnv
from gymnasium_env.envs.variable_maze_env import VariableMazeEnv

from tqdm import tqdm

class OffPolicyTrainer():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        pass

    def train(self,n_episodes:int):
        # reset the environment to get the first observation
        done = False

        for episode in tqdm(range(n_episodes)):
            obs, _ = self.env.reset()
            done = False
            cumulative = 0

            # play one episode
            while not done:
                action = self.agent.get_action(self.env, obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                cumulative += reward

                # update the agent
                self.agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            print(f'episode {episode} size {self.env.env.current_shape} cumulative reward {cumulative}')
            self.agent.decay_epsilon()
    
    def test(self, num_mazes:int):
        win = 0

        for _ in range(num_mazes):
            if isinstance(self.env.env, MazeEnv):
                self.env.env.update_maze()
            elif isinstance(self.env.env, VariableMazeEnv):
                self.env.env.update_maze(False)

            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.agent.get_action(self.env, obs)
                next_obs, reward, truncated, terminated, _ = self.env.step(action)
                total_reward += reward
                if terminated:
                    win += 1
                    done = True
                else:
                    done = truncated
                obs = next_obs
        
            print(f'total reward {total_reward}')
        
        print(f'winrate {(win / num_mazes)*100} %')

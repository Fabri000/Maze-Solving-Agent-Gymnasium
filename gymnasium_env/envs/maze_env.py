from enum import Enum
from typing import Optional

import numpy as np

from gymnasium_env.envs.maze_view import MazeView

import gymnasium as gym
from gymnasium import spaces

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],"render_fps": 4}

    ACTIONS = {           
            0: np.array([1, 0]),  # down
            1: np.array([-1, 0]),  # up
            2: np.array([0, 1]),  # right
            3: np.array([0, -1]),  # left
        }


    def __init__(self, maze_map, start_pos, goal_pos, render_mode = "human"):
        super().__init__()

        self.render_mode = render_mode
        self.maze_map = maze_map
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        if self.render_mode == "human":
            self.maze_view = MazeView(self.maze_map,self.start_pos,self.goal_pos,(len(maze_map),len(maze_map[1])))

        self._agent_location = np.array(self.start_pos, dtype=np.int32)
        self._target_location = np.array(self.goal_pos, dtype=np.int32)

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(0,len(maze_map)*len(maze_map[0]),shape=(2,),dtype=int),
                "target": gym.spaces.Box(0,len(maze_map)*len(maze_map[0]),shape=(2,),dtype=int)
            }
        )
        self.action_space = gym.spaces.Discrete(4)

        self.step_count = len(maze_map)*len(maze_map[0])
        
        self.reset()
    
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location-self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Choose the agent's location uniformly at random
        self._agent_location = np.array(self.start_pos,dtype=np.int32)
        self.maze_view._reset_agent()

        self.step_count = 2*len(self.maze_map)*len(self.maze_map[0])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        reward =  0
        terminated = False
        done = False

        self.step_count -= 1

        moved = self.maze_view.move_agent(MazeEnv.ACTIONS[action])
        
        if moved:
            self._agent_location = np.array(self.maze_view._agent_position,dtype=np.int32)

            if np.array_equal(self._agent_location,self._target_location):
                reward = 1
                terminated = True
            else:
                reward = -0.05
        else:
            reward = -0.25
        
        if self.step_count == 0:
            done = True

        observation = self._get_obs()
        info = self._get_info()

        if done or terminated:
            self.reset()

        return observation, reward, terminated, done, info
    
    def render(self,mode="human",close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)

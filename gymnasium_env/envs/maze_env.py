from enum import Enum
from typing import Optional

import numpy as np

from gymnasium_env.envs.maze_view import MazeView
from lib.a_star import astar_limited_partial

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


    def __init__(self, maze_map,start_pos,goal_pos, render_mode = "human"):
        super().__init__()

        self.render_mode = render_mode
        self.maze_map = maze_map
        self._start_pos =start_pos
        self._goal_pos = goal_pos

        if self.render_mode == "human":
            self.maze_view = MazeView(self.maze_map,self._start_pos,self._goal_pos,(len(maze_map),len(maze_map[1])))

        self._agent_location = np.array(self._start_pos, dtype=np.int32)
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(0,len(maze_map)*len(maze_map[0]),shape=(2,),dtype=int),
                "target": gym.spaces.Box(0,len(maze_map)*len(maze_map[0]),shape=(2,),dtype=int),
                "best next": gym.spaces.Box(0,len(maze_map)*len(maze_map[0]),shape=(2,),dtype=int)
            }
        )
        self.action_space = gym.spaces.Discrete(4)

        self.step_count = len(maze_map)*len(maze_map[0])
        
        self.reset()

    def _find_best_next_cell(self,agent_pos):
        paths = []
        for dir in MazeEnv.ACTIONS:
            next_pos = tuple(agent_pos + MazeEnv.ACTIONS[dir])
            if 0<next_pos[0]<len(self.maze_map) and 0<next_pos[1]<len(self.maze_map[0]) and self.maze_map[next_pos[0]][next_pos[1]]:
                paths.append(astar_limited_partial(self.maze_map,next_pos,self._goal_pos,max_depth=min(len(self.maze_map),len(self.maze_map[1]))))
        best_dist = len(self.maze_map)*len(self.maze_map[0])
        best_path = None
        for path in paths:
            dist_to_goal = len(astar_limited_partial(self.maze_map,path[-1],self._goal_pos,max_depth=len(self.maze_map)*len(self.maze_map[1])))
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_path = path
        return best_path[0]

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location,"best next":self._find_best_next_cell(self._agent_location)}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location-self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Choose the agent's location uniformly at random
        self._agent_location = np.array(self._start_pos,dtype=np.int32)
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

        prev_pos = self._agent_location
        moved = self.maze_view.move_agent(MazeEnv.ACTIONS[action])
        
        if moved:
            self._agent_location = np.array(self.maze_view._agent_position,dtype=np.int32)

            if np.array_equal(self._agent_location,self._target_location):

                print("Win")
                reward = 10
                terminated = True
            else:
                new_dist = len(astar_limited_partial(self.maze_map,tuple(self._agent_location),tuple(self._target_location)))
                old_dist = len(astar_limited_partial(self.maze_map,tuple(prev_pos),tuple(self._target_location)))
                reward = (old_dist - new_dist) * 0.2 - 0.05
        else:
            reward = -0.3
        
        if self.step_count == 0:
            reward = -0.5
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

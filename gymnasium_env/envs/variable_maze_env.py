from typing import Optional

import random
import math
import numpy as np

from lib.maze_view import MazeView
from lib.maze_generator import gen_maze
from lib.a_star import astar_limited_partial

import gymnasium as gym
from gymnasium import spaces

class VariableMazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],"render_fps": 4}
    START_SHAPE = (7,7)

    ACTIONS = {           
            0: np.array([1, 0]),  # down
            1: np.array([-1, 0]),  # up
            2: np.array([0, 1]),  # right
            3: np.array([0, -1]),  # left
        }
    
    def __init__(self,max_shape:tuple[int,int],render_mode = "human"):
        self.render_mode = render_mode
        self.max_shape = max_shape

        self.maze_shape = VariableMazeEnv.START_SHAPE
        self._start_pos , self.maze_map,= gen_maze(self.maze_shape)
        goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]

        self._agent_location = np.array(self._start_pos, dtype=np.int32)
        self._target_location = np.array(goal_pos, dtype=np.int32)

        if self.render_mode == "human":
            self.maze_view = MazeView(self.maze_map,self._start_pos,self._target_location,self.maze_shape)
        

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(low=np.array([0,0]),high=np.array(self.max_shape)),
                "target": gym.spaces.Box(low=np.array([0,0]),high=np.array(self.max_shape)),
                "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(4)
        
        self.min_cum_rew = - self.maze_shape[0]
        self.cum_rew = 0
        self.visited_cell= []
        self.consecutive_invalid_moves = 0

        self.mazes = [[self._start_pos, self.maze_shape, self.maze_map]]
        self.next = 0
        self.reset()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location,"best dir": self._agent_location - self._find_best_next_cell(self._agent_location)}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location-self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._agent_location = np.array(self._start_pos,dtype=np.int32)
        self.maze_view._reset_agent()

        observation = self._get_obs()
        info = self._get_info()

        self.cum_rew = 0
        self.consecutive_invalid_moves = 0
        self.visited_cell= []

        return observation, info
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        prev_pos = self._agent_location
        moved = self.maze_view.move_agent(VariableMazeEnv.ACTIONS[action])

        if moved:
            self._agent_location = np.array(self.maze_view._agent_position, dtype=np.int32)
            current_cell = tuple(self._agent_location)
            self.consecutive_invalid_moves = 0

            if current_cell not in self.visited_cell:
                if np.array_equal(self._agent_location, self._target_location):
                    reward = 1
                    terminated = True
                else:
                    new_dist = len(astar_limited_partial(self.maze_map, current_cell, tuple(self._target_location)))
                    old_dist = len(astar_limited_partial(self.maze_map, tuple(prev_pos), tuple(self._target_location)))
                    reward = (old_dist - new_dist) * 0.5
            else:
                reward = -1 + math.exp(- 0.1 * (self.visited_cell.count(current_cell)))

            self.visited_cell.append(current_cell)
        else:
            self.consecutive_invalid_moves += 1
            reward = -1 + math.exp(- 0.15 * (self.consecutive_invalid_moves))

        observation = self._get_obs()
        info = self._get_info()
        
        self.cum_rew += reward
        if self.cum_rew < self.min_cum_rew:
            truncated = True

        if truncated or terminated:
            self.reset()

        return observation, reward, truncated, terminated, info

    def render(self,mode="human",close=False):
        if close:
            self.maze_view.quit_game()
        return self.maze_view.update(mode)

    def get_maze_shape(self):
        return self.maze_shape
    
    def get_max_shape(self):
        return self.max_shape

    def update_maze(self):
        shape = tuple(a+b for a,b in zip(self.maze_shape,(2,2)))
        if shape <= self.max_shape:
            self.maze_shape = shape
            self.min_cum_rew = - min(self.maze_shape[0],self.max_shape[1])
            
            self._start_pos , self.maze_map= gen_maze(self.maze_shape)
            goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]
            self._target_location = np.array(goal_pos, dtype=np.int32)
            self.mazes.append([self._start_pos,self.maze_shape,self.maze_map])
            
            self.maze_view.update_maze(self.maze_map,self._start_pos,self._target_location,self.maze_shape)
            self.reset()

        else:
            random.shuffle(self.mazes)
    
    def update_visited_maze(self, remove= True):
        self._start_pos, self.maze_shape,self.maze_map = self.mazes[self.next]

        if remove:
            self.mazes.remove([self._start_pos,self.maze_shape,self.maze_map])
        else:
            self.next +=1
        
        self.min_cum_rew = - min(self.maze_shape[0],self.maze_shape[1])
        goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]
        self._target_location = np.array(goal_pos, dtype=np.int32)
        
        self.maze_view.update_maze(self.maze_map,self._start_pos,self._target_location,self.maze_shape)
        self.reset()

    def _find_best_next_cell(self,agent_pos):
        paths = []
        for dir in VariableMazeEnv.ACTIONS:
            next_pos = tuple(agent_pos + VariableMazeEnv.ACTIONS[dir])
            if 0<next_pos[0]<len(self.maze_map) and 0<next_pos[1]<len(self.maze_map[0]) and self.maze_map[next_pos[0]][next_pos[1]]:
                paths.append(astar_limited_partial(self.maze_map,next_pos,tuple(self._target_location.tolist()),max_depth=min(len(self.maze_map),len(self.maze_map[1]))))
        best_dist = len(self.maze_map)*len(self.maze_map[0])
        best_path = None
        for path in paths:
            dist_to_goal = len(astar_limited_partial(self.maze_map,path[-1],tuple(self._target_location.tolist()),max_depth=len(self.maze_map)*len(self.maze_map[1])))
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_path = path
        return best_path[0]
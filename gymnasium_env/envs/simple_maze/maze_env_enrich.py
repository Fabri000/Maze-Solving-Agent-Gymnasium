from typing import Optional

import math
import numpy as np

from lib.maze_view import MazeViewTemplate
from lib.a_star_algos.a_star import astar_limited_partial
from lib.maze_generation import gen_maze
from lib.maze_handler import extract_submaze,get_mask_tensor

import gymnasium as gym
from gymnasium import spaces

class EnrichMazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],"render_fps": 4}

    ACTIONS = {           
            0: np.array([1, 0]),  # down
            1: np.array([-1, 0]),  # up
            2: np.array([0, 1]),  # right
            3: np.array([0, -1]),  # left
        }


    def __init__(self, maze_map,start_pos,goal_pos,encoder,render_mode = "human"):
        super().__init__()
        self.encoder = encoder

        self.render_mode = render_mode
        self.maze_map = maze_map
        self.maze_shape = (len(maze_map),len(maze_map[0]))
        self._start_pos =start_pos
        self._goal_pos = goal_pos

        if self.render_mode == "human":
            self.maze_view = MazeViewTemplate(self.maze_map,self._start_pos,self._goal_pos,(len(maze_map),len(maze_map[1])))

        self._agent_location = np.array(self._start_pos, dtype=np.int32)
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(0,self.maze_shape[0]*self.maze_shape[1],shape=(2,),dtype=int),
                "target": gym.spaces.Box(0,self.maze_shape[0]*self.maze_shape[1],shape=(2,),dtype=int),
                "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int),
                "window_feature": gym.spaces.Box(-1,1,shape=(72,),dtype=float),
            }
        )
        self.action_space = gym.spaces.Discrete(4)

        self.min_cum_rew = - min(self.maze_shape[0],self.maze_shape[1])
        self.cum_rew = 0
        self.visited_cell= []
        self.consecutive_invalid_moves = 0

        self.mazes = [[self._start_pos, self.maze_map]]
        self.next = 0
        self.reset()

    def _find_best_next_cell(self,agent_pos):
        paths = []
        for dir in EnrichMazeEnv.ACTIONS:
            next_pos = tuple(agent_pos + EnrichMazeEnv.ACTIONS[dir])
            if 0<next_pos[0]<self.maze_shape[0] and 0<next_pos[1]<self.maze_shape[1] and self.maze_map[next_pos[0]][next_pos[1]]:
                paths.append(astar_limited_partial(self.maze_map,next_pos,self._goal_pos,max_depth=min(self.maze_shape[0],self.maze_shape[1])))
        best_dist = self.maze_shape[0]*self.maze_shape[1]
        best_path = None
        for path in paths:
            dist_to_goal = len(astar_limited_partial(self.maze_map,path[-1],self._goal_pos,max_depth=self.maze_shape[0]*self.maze_shape[1]))
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_path = path
        return best_path[0]

    def _get_collitions(self,agent_pos):
        free_cell = []
        for dir in EnrichMazeEnv.ACTIONS:
            next_pos = tuple(agent_pos + EnrichMazeEnv.ACTIONS[dir])
            if 0<next_pos[0]<self.maze_shape[0] and 0<next_pos[1]<self.maze_shape[1]:
                if self.maze_map[next_pos[0]][next_pos[1]]:
                    free_cell.append(0)
                else:
                    free_cell.append(1)
        return free_cell

    def _get_obs(self):
        sub_maze = extract_submaze(self.maze_map,self._agent_location,15)
        feature = self.encoder(get_mask_tensor(sub_maze)).flatten().detach()
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)

        return {"agent": self._agent_location, 
                "target": self._target_location,
                "best dir": self._agent_location - self._find_best_next_cell(self._agent_location),
                "window_feature": feature
                }
    
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

        self.cum_rew=0
        self.consecutive_invalid_moves = 0
        self.visited_cell= []

        return observation, info
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        prev_pos = self._agent_location
        moved = self.maze_view.move_agent(EnrichMazeEnv.ACTIONS[action])

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
                reward
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
    
    def update_maze(self):
        self._start_pos , self.maze_map = gen_maze(self.maze_shape)

        self._goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.mazes.append([self._start_pos,self.maze_map])

        self.maze_view.update_maze(self.maze_map,self._start_pos,self._goal_pos,self.maze_shape)
        self.reset()
    
    def update_visited_maze(self, remove= True):
        self._start_pos, self.maze_map = self.mazes[self.next]

        if remove:
            self.mazes.remove([self._start_pos,self.maze_map])
        else:
            self.next+=1

        self._goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]    
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.maze_view.update_maze(self.maze_map,self._start_pos,self._goal_pos,self.maze_shape)
        self.reset()
    
    def get_maze_shape(self):
        return self.maze_shape
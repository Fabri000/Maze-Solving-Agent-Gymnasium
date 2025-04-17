import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn

from gymnasium_env.envs.base_maze_env import BaseConstantSizeEnv
from lib.a_star_algos.a_star import astar_limited_partial
from lib.maze_difficulty_evaluation.metrics_calculator import MetricsCalculator
from lib.maze_handler import extract_submaze, get_mask_tensor,get_direction_mask
from lib.maze_view import SimpleMazeView


class SimpleMazeEnv(BaseConstantSizeEnv):
    """
    A class representing an environment for a maze game.
    """

    def __init__(self,maze_shape:tuple[int,int],render_mode:str="human"):
        """
        Initialize the maze environment.
        Args:
            maze_shape (tuple): the size of the maze.
            render_mode (str): the rendering mode. Default: "human"."""
        
        start_pos,goal_pos,maze_map = self.generate_maze(maze_shape)

        super(SimpleMazeEnv, self).__init__(maze_map,start_pos,goal_pos,maze_shape)
        self.set_max_steps()

        if render_mode == "human":
            self.maze_view = SimpleMazeView(maze_map,start_pos,goal_pos,maze_shape)

        self.mazes.append([self._start_pos,self.maze_map])

        self.reset()
        
    def next_cell(self, agent_pos, dir):
        return tuple(agent_pos + SimpleMazeEnv.ACTIONS[dir])
    
    def get_mask_direction(self,probs = False):
        """
        Get the mask for the direction that the agent can move."""
        mask = get_direction_mask(self.maze_map,self._agent_location)
        if probs and len(self.visited_cell)>1:
            mask = mask.astype(np.float32)
            previous = tuple(self.visited_cell[-2]-self._agent_location ) 
            dir = [(1,0),(-1,0), (0,1), (0,-1)].index(tuple(previous))
            mask[dir] = 0.25
        return mask

    def set_max_steps(self):
        """
        Set the maximum steps that the agent can take in the episode
        """
        path = self.find_path(self._start_pos)
        factor = MetricsCalculator(self.maze_map, len(path)).calculate_L(path)
        self.max_steps_taken = math.ceil((((self.maze_shape[0]-1) * (self.maze_shape[1]-1)) - 1) * factor)
    
    def valid_cell(self, pos):
        """
        Check if the cell is valid.
        Args:
            pos (tuple): the position of the cell.
        Returns:
            bool: whether the cell is valid.
        """
        return 0<pos[0]<self.maze_shape[0] and 0<pos[1]<self.maze_shape[1] and self.maze_map[pos[0]][pos[1]]!=0
    
    def find_path(self,source:tuple[int,int],max_depth:int=1e6):
        """
        Find the path to the goal.
        Args:  
            source (tuple): the start position for the search.
            max_depth (int): the maximum depth of the search
        Returns:
            list: the path to the goal.
        """
        return astar_limited_partial(self.maze_map,source,tuple(self._target_location),max_depth=max_depth)

    def update_maze(self):
        """
        Update the maze.
        """
        self._start_pos,goal_pos,self.maze_map= self.generate_maze(self.maze_shape)

        self._target_location = np.array(goal_pos, dtype=np.int32)

        self.set_max_steps()

        self.mazes.append([self._start_pos,self.maze_map])

        self.maze_view.update_maze(self.maze_map,self._start_pos,tuple(self._target_location),self.maze_shape)
        self.reset()
    
    def update_visited_maze(self, remove: bool = True):
        """
        Update the visited maze.
        Args:
            remove (bool): whether to remove the visited cells. Default: True.
        """
        self._start_pos, self.maze_map = self.mazes[self.next]
       
        goal_pos =  [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]

        if remove:
            self.mazes.remove([self._start_pos,self.maze_map])
        else:
            self.next+=1

        self._target_location = np.array(goal_pos, dtype=np.int32)
        
        self.set_max_steps()

        self.maze_view.update_maze(self.maze_map,self._start_pos,tuple(self._target_location),self.maze_shape)
        self.reset()
    
    def update_new_maze(self, shape:tuple[int,int]=None):
        if shape is not None:
            self.maze_shape = shape
        self._start_pos, goal_pos, self.maze_map = self.generate_maze(self.maze_shape)
        self._target_location = np.array(goal_pos, dtype=np.int32)
        
        self.set_max_steps()

        self.maze_view.update_maze(self.maze_map,self._start_pos,self._target_location,self.maze_shape)
        self.reset()
    
class SimpleEnrichMazeEnv(SimpleMazeEnv):
    WINDOW_DIM = 15
    def __init__(self,maze_shape:tuple[int,int],render_mode:str="human"):
        """
        Initialize the maze environment.
        Args:
            maze_shape (tuple): the size of the maze.
            render_mode (str): the rendering mode. Default: "human".
        """

        super(SimpleEnrichMazeEnv, self).__init__(maze_shape,render_mode)
        

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(0,1,shape=(2,),dtype=int),
                "target": gym.spaces.Box(0,1,shape=(2,),dtype=int),
                "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int),
                "window": gym.spaces.Box(-1,1,shape=(4,SimpleEnrichMazeEnv.WINDOW_DIM,SimpleEnrichMazeEnv.WINDOW_DIM),dtype=float)
            }
        )

    def _get_obs(self):
        sub_maze, non_visited ,position = extract_submaze(self.maze_map,self.non_visited,self._agent_location,SimpleEnrichMazeEnv.WINDOW_DIM)
        mask = get_mask_tensor(sub_maze,non_visited,position)

        return {"agent": self._agent_location / self.maze_shape,
                "target": self._target_location / self.maze_shape,
                "best dir": self._agent_location - self._find_best_next_cell(self._agent_location),
                "window": mask}   
    
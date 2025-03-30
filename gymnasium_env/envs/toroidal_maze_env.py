import math
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from gymnasium_env.envs.base_maze_env import BaseConstantSizeEnv
from lib.maze_generation import gen_maze_no_border
from lib.maze_handler import extract_submaze_toroid, get_mask_tensor
from lib.maze_difficulty_evaluation.metrics_calculator import MetricsCalculator
from lib.maze_view import ToroidalMazeView
from lib.a_star_algos.a_star_tor import astar_limited_partial

class ToroidalMazeEnv(BaseConstantSizeEnv):
    """
    A class representing an environment for a maze game where the maze has a toroidal structure.
    """
    def __init__(self,maze_shape:tuple[int,int],render_mode:str="human"):
        """
        Initialize the maze environment.
        Args:
            maze_shape (tuple): the size of the maze.
            render_mode (str): the rendering mode. Default: "human".
        """
        self.render_mode = render_mode

        start_pos, goal_pos, maze_map = self.generate_maze(maze_shape)

        super(ToroidalMazeEnv, self).__init__(maze_map,start_pos,goal_pos,maze_shape)
        self.set_max_steps()

        if render_mode == "human":
            self.maze_view = ToroidalMazeView(self.maze_map,self._start_pos,self._target_location,self.maze_shape)

        self.mazes.append([self._start_pos,self.maze_map])

        self.reset()
    
    def generate_maze(self,maze_shape:tuple[int,int]):
        """
        Generate a new maze.
        Args:
            maze_shape (tuple): the shape of the maze.
        """
        start_pos,goal_pos,maze_map,difficulty = gen_maze_no_border(maze_shape,ToroidalMazeEnv.ALGORITHM)
        
        for _ in range(5):
            tmp_start_pos,tmp_goal_pos,tmp_maze_map, tmp_difficulty = gen_maze_no_border(maze_shape,ToroidalMazeEnv.ALGORITHM)
            if tmp_difficulty < difficulty:
                start_pos,goal_pos,maze_map = tmp_start_pos,tmp_goal_pos,tmp_maze_map
                difficulty = tmp_difficulty
        
        return start_pos,goal_pos,maze_map


    def set_max_steps(self):
        """
        Set the maximum steps that the agent can take in the episode
        """
        path = self.find_path(self._start_pos)
        factor = MetricsCalculator(self.maze_map, len(path)).calculate_L(path)
        self.max_steps_taken = math.ceil((((self.maze_shape[0]-1) * (self.maze_shape[1]-1)) - 1) * factor)

    def next_cell(self, agent_pos, dir):
        next_cell = tuple(agent_pos + ToroidalMazeEnv.ACTIONS[dir])
        return (next_cell[0] % self.maze_shape[0], next_cell[1] % self.maze_shape[1])
    
    def valid_cell(self, pos):
        """
        Check if the cell is valid.
        """
        return self.maze_map[pos[0]][pos[1]]
    
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
        Update the game maze.
        """
        self._start_pos,goal_pos,self.maze_map = self.generate_maze(self.maze_shape)
        self._target_location = np.array(goal_pos, dtype=np.int32)

        self.set_max_steps()

        self.mazes.append([self._start_pos,self.maze_map])

        self.maze_view.update_maze(self.maze_map,self._start_pos,tuple(self._target_location),self.maze_shape)
        self.reset()

    def update_visited_maze(self, remove: bool = True):
        """
        Update the game with a maze that was already learned at training time.
        Args:
            remove (bool): whether to remove the visited maze from the list of learned maze. Default: True.
        """
        self._start_pos, self.maze_map = self.mazes[self.next]
        goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]

        if remove:
            self.mazes.remove([self._start_pos,self.maze_map])
        else:
            self.next+=1

        self._target_location = np.array(goal_pos, dtype=np.int32)
        self.set_max_steps()

        self.maze_view.update_maze(self.maze_map,self._start_pos,tuple(self._target_location),self.maze_shape)
        self.reset()
    
    def update_new_maze(self):
        self._start_pos, goal_pos, self.maze_map =  self.generate_maze(self.maze_shape)
        self._target_location = np.array(goal_pos, dtype=np.int32)
        
        self.set_max_steps()

        self.maze_view.update_maze(self.maze_map,self._start_pos,self._target_location,self.maze_shape)
        self.reset()

class ToroidalEnrichMazeEnv(ToroidalMazeEnv):
        """
        A class representing an environment for a maze game where the maze has a toroidal structure and
        can have variable sizes. It adds to the observation feature extracted by a Convolutional Encoder
        on a fixed size window.
        """
        def __init__(self,max_shape:tuple[int,int],render_mode:str="human"):
            super(ToroidalEnrichMazeEnv, self).__init__(max_shape,render_mode)

            self.observation_space = spaces.Dict(
                {
                    "agent": gym.spaces.Box(low = np.array([0,0]), high= np.array(self.maze_shape),dtype=int),
                    "target": gym.spaces.Box(low = np.array([0,0]), high= np.array(self.maze_shape),shape=(2,),dtype=int),
                    "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int),
                    "window": gym.spaces.Box(-1,1,shape=(4,15,15),dtype=float)
                }
            )
        
        def _get_obs(self):
            sub_maze,position = extract_submaze_toroid(self.maze_map,self._agent_location,15)
            mask = get_mask_tensor(sub_maze,position)

            return {"agent": self._agent_location,
                    "target": self._target_location,
                    "best dir": self._agent_location - self._find_best_next_cell(self._agent_location),
                    "window": mask
            }
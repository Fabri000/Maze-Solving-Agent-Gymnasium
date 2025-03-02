import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from gymnasium_env.envs.base_maze_env import BaseVariableSizeEnv
from lib.maze_generation import gen_maze_no_border
from lib.maze_handler import extract_submaze_toroid, get_mask_tensor
from lib.maze_view import ToroidalMazeView
from lib.a_star_algos.a_star_tor import astar_limited_partial


class ToroidalVariableMazeEnv(BaseVariableSizeEnv):
    """
    A class representing an environment for a maze game where the maze has a toroidal structure and
    can have variable sizes.
    """
    START_SHAPE = (15,15)

    def __init__(self,max_shape:tuple[int,int],render_mode:str="human"):
        """
        Initialize the maze environment.
        Args:
            max_shape (tuple): the maximum size of the maze.
            render_mode (str): the rendering mode. Default: "human".
        """
        self.max_shape = max_shape
        self.render_mode = render_mode
        
        maze_shape = ToroidalVariableMazeEnv.START_SHAPE
        start_pos, maze_map = gen_maze_no_border(maze_shape)
        goal_pos = [(r, c) for r in range(maze_shape[0]) for c in range(maze_shape[1]) if maze_map[r][c] == 2][-1]

        super(ToroidalVariableMazeEnv, self).__init__(maze_map,start_pos, goal_pos, maze_shape)

        if render_mode == "human":
            self.maze_view = ToroidalMazeView(self.maze_map,self._start_pos,self._target_location,self.maze_shape)

        self.mazes.append([self._start_pos,self.maze_shape,self.maze_map])

        self.reset()
    
    def get_max_shape(self):
        """ 
        Get the maximum shape of the maze.
        Returns:
            tuple: the maximum shape of
        """
        return self.max_shape

    def next_cell(self, agent_pos, dir):
        next_cell = tuple(agent_pos + ToroidalVariableMazeEnv.ACTIONS[dir])
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
        shape = tuple(a+b for a,b in zip(self.maze_shape,(2,2)))
        if shape <= self.max_shape:
            self.maze_shape = shape
            self.min_cum_rew = - min(self.maze_shape[0],self.maze_shape[1])
            
            self._start_pos , self.maze_map = gen_maze_no_border(self.maze_shape)

            goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]
            self._target_location = np.array(goal_pos, dtype=np.int32)

            self.mazes.append([self._start_pos,self.maze_shape,self.maze_map])

            self.maze_view.update_maze(self.maze_map,self._start_pos,self._target_location,self.maze_shape)
            self.reset()
        else:
            random.shuffle(self.mazes)
    
    def update_visited_maze(self, remove: bool = True):
        """
        Update the game with a maze that was already learned at training time.
        Args:
            remove (bool): whether to remove the visited maze from the list of learned maze. Default: True.
        """
        self._start_pos,self.maze_shape,self.maze_map = self.mazes[self.next]

        if remove:
            self.mazes.remove([self._start_pos,self.maze_map])
        else:
            self.next+=1

        self.min_cum_rew = - min(self.maze_shape[0],self.maze_shape[1])
        self._goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]    
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.maze_view.update_maze(self.maze_map,self._start_pos,self._goal_pos,self.maze_shape)
        self.reset()
    
class ToroidalEnrichVariableMazeEnv(ToroidalVariableMazeEnv):
        """
        A class representing an environment for a maze game where the maze has a toroidal structure and
        can have variable sizes. It adds to the observation feature extracted by a Convolutional Encoder
        on a fixed size window.
        """
        def __init__(self,max_shape:tuple[int,int],encoder:nn.Sequential,render_mode:str="human"):
            self.encoder = encoder
            super(ToroidalEnrichVariableMazeEnv, self).__init__(max_shape,render_mode)

            self.observation_space = spaces.Dict(
                {
                    "agent": gym.spaces.Box(0,self.maze_shape[0]*self.maze_shape[1],shape=(2,),dtype=int),
                    "target": gym.spaces.Box(0,self.maze_shape[0]*self.maze_shape[1],shape=(2,),dtype=int),
                    "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int),
                    "window_feature": gym.spaces.Box(-1,1,shape=(72,),dtype=float),
                }
            )
        
        def _get_obs(self):
            sub_maze = extract_submaze_toroid(self.maze_map,self._agent_location,15)
            mask = get_mask_tensor(sub_maze)
            feature = self.encoder(mask).flatten().detach()
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)

            return {"agent": self._agent_location, 
                    "target": self._target_location,
                    "best dir": self._agent_location - self._find_best_next_cell(self._agent_location),
                    "window_feature": feature
            }
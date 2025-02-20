import math
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseMazeEnv(gym.Env):
    """
    A base class representing an environment for a maze game.
    """

    metadata = {'render.modes': ['human', 'rgb_array'],"render_fps": 4}

    ACTIONS = {           
            0: np.array([1, 0]),  # down
            1: np.array([-1, 0]),  # up
            2: np.array([0, 1]),  # right
            3: np.array([0, -1]),  # left
        }

    def __init__(self,maze_map,start_pos:tuple[int,int],goal_pos:tuple[int,int],maze_shape:tuple[int,int]):
        """
        Initialize the maze environment.
        Args:
            maze_map (array): the array representation of the maze where 0 represent walls, 1 the walkable tiles and 2 the goal.
            start_pos (tuple): the starting position of the agent.
            goal_pos (tuple): the goal position of the agent.
            maze_shape (tuple): the shape of the maze.
        """
        self.maze_map = maze_map

        self.maze_shape = maze_shape
        self._start_pos = start_pos
        self._goal_pos = goal_pos

        self._agent_location = np.array(self._start_pos, dtype=np.int32)
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(0,self.maze_shape[0]*self.maze_shape[1],shape=(2,),dtype=int),
                "target": gym.spaces.Box(0,self.maze_shape[0]*self.maze_shape[1],shape=(2,),dtype=int),
                "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(4)
        
        self.maze_view = None

        self.min_cum_rew = - min(self.maze_shape[0],self.maze_shape[1])
        self.cum_rew = 0
        self.visited_cell = []
        self.consecutive_invalid_moves = 0
        self.mazes = []
        self.next = 0
    
    def get_maze_shape(self):
        """
        Get the shape of the maze.
        Returns:
            tuple: the shape of the maze.
        """
        return self.maze_shape
    
    def _get_obs(self):
        """
        Get the observation of the environment.
        Returns:
            dict: the observation of the environment.
        """
        return {"agent": self._agent_location, "target": self._target_location,"best dir": self._agent_location - self._find_best_next_cell(self._agent_location)}
    
    def _get_info(self):
        """
        Get the information of the environment.
        Returns:
            dict: the information of the environment.
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location-self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment.
        Args:
            seed (int): the seed for the random number generator. Default: None.
            options (dict): the options for the environment. Default: None.
        Returns:
            tuple: the observation and information of the environment.
        """

        self._agent_location = np.array(self._start_pos,dtype=np.int32)
        self.maze_view._reset_agent()

        observation = self._get_obs()
        info = self._get_info()

        self.cum_rew = 0
        self.consecutive_invalid_moves = 0
        self.visited_cell= []

        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action (int): the action to take.
        Returns:
            tuple: the observation, reward, termination status, information of the environment.
        """
        reward = 0
        terminated = False
        truncated = False

        prev_pos = self._agent_location
        moved = self.maze_view.move_agent(BaseMazeEnv.ACTIONS[action])

        if moved:
            self._agent_location = np.array(self.maze_view._agent_position, dtype=np.int32)
            current_cell = tuple(self._agent_location)
            self.consecutive_invalid_moves = 0

            if current_cell not in self.visited_cell:
                if np.array_equal(self._agent_location, self._target_location):
                    reward = 1
                    terminated = True
                else:
                    new_dist = len(self.find_path(current_cell))
                    old_dist = len(self.find_path(tuple(prev_pos)))
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
        """
        Render the environment.
        Args:
            mode (str): the rendering mode. Default: "human".
            close (bool): whether to close the environment. Default: False."""
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)
    
    def _find_best_next_cell(self, agent_pos):
        """
        Find the best next cell for the agent.
        Args:
            agent_pos (tuple): the position of the agent.
        Returns:
            tuple: the best next cell for the agent.
        """
        paths = []
        for dir in BaseMazeEnv.ACTIONS:
            next_pos = self.next_cell(agent_pos,dir)#depends on type of maze
            if self.valid_cell(next_pos):
                paths.append(self.find_path(next_pos,max_depth=min(self.maze_shape[0],self.maze_shape[1])))
        best_dist = self.maze_shape[0]*self.maze_shape[1]
        best_path = None
        for path in paths:
            dist_to_goal = len(self.find_path(path[-1]))
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_path = path
        return best_path[0]
    
    def next_cell(self, agent_pos:tuple[int,int],dir:int):
        """
        Get the next cell for the agent.
        Args:
            agent_pos (tuple): the position of the agent.
            dir (int): the direction of the agent.
        Returns:
            tuple: the next cell for the agent.
        """
        
        raise NotImplementedError("Subclasses must implement next_cell")
    
    def valid_cell(self, pos:tuple[int,int]):
        """
        Check if the next cell is valid.
        Args:
            pos (tuple): position in the maze.
        Returns:
            bool: if the cell is valid
        """
        raise NotImplementedError("Subclasses must implement valid_cell")

    def find_path(self,source:tuple[int,int],max_depth:int=1e6):
        """
        Find the path to the goal.
        Args:  
            source (tuple): the start position for the search.
            max_depth (int): the maximum depth of the search
        Returns:
            list: the path to the goal.
        """
        return NotImplementedError("Subclasses must implement find_path")

    def update_visited_maze(self, remove: bool = True):
        """
        Update the game with a maze that was already learned at training time.
        Args:
            remove (bool): whether to remove the visited maze from the list of learned maze. Default: True.
        """
        raise NotImplementedError("Subclasses must implement update_visited_maze")
    
    def update_maze(self):
        """
        Update the maze.
        """
        raise NotImplementedError("Subclasses must implement update_maze")
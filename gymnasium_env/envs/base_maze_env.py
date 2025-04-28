import math
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation
from lib.maze_generation import gen_maze

class BaseMazeEnv(gym.Env):
    """
    A base class representing an environment for a maze game.
    """

    metadata = {'render.modes': ['human', 'rgb_array'],"render_fps": 4}

    ALGORITHM = "r-prim"  # Default algorithm for maze generation

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

        self.non_visited = (np.array(self.maze_map) !=0).astype(np.int32).tolist()
        self.non_visited[start_pos[0]][start_pos[1]] = 0

        self._start_pos = start_pos

        self._agent_location = np.array(self._start_pos, dtype=np.int32)
        self._target_location = np.array(goal_pos, dtype=np.int32)

        self.observation_space = None
        self.action_space = gym.spaces.Discrete(4)
        
        self.maze_view = None

        self.max_steps_taken = 0
        self.cum_rew = 0
        self.visited_cell = []
        self.consecutive_invalid_moves = 0
        self.mazes = []
        self.next = 0

    def set_algorithm(self,algorithm:str):
        BaseMazeEnv.ALGORITHM = algorithm
    
    def get_algorithm(self):
        return BaseMazeEnv.ALGORITHM
    

    def get_mask_direction(self,probs = False):
        """
        Get the mask for the direction that the agent can move."""
        return NotImplementedError("Subclasses must implement get_mask_direction")
    
    def set_max_steps(self):
        """
        Set the maximum steps that the agent can take in the episode
        """
        return NotImplementedError("Subclasses must implement set_max_steps")
    
    def generate_maze(self,maze_shape:tuple[int,int]):
        """
        Generate a new maze.
        Args:
            maze_shape (tuple): the shape of the maze.
        """
        start_pos,goal_pos,maze_map = gen_maze(maze_shape,BaseMazeEnv.ALGORITHM)
        difficulty = ComplexityEvaluation(maze_map,start_pos,goal_pos).difficulty_of_maze()

        for _ in range(5):
            tmp_start_pos,tmp_goal_pos,tmp_maze_map = gen_maze(maze_shape,BaseMazeEnv.ALGORITHM)
            tmp_difficulty = ComplexityEvaluation(tmp_maze_map,tmp_start_pos,tmp_goal_pos).difficulty_of_maze() 
            if tmp_difficulty < difficulty:
                start_pos = tmp_start_pos
                goal_pos = tmp_goal_pos
                maze_map = tmp_maze_map

                difficulty = tmp_difficulty
        
        return start_pos,goal_pos,maze_map

    def get_maze_difficulty(self):
        """
        Get the difficulty of the maze.
        Returns:
            float: the difficulty of the maze.
        """
        return ComplexityEvaluation(self.maze_map,self._start_pos,tuple(self._target_location)).difficulty_of_maze()

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
        return {"agent": self._agent_location,"target":self._target_location, "best dir": self._agent_location - self._find_best_next_cell(self._agent_location)}
    
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
        start_pos = tuple(self._agent_location)
        self.non_visited = (np.array(self.maze_map) !=0).astype(np.int32).tolist()
        self.non_visited[start_pos[0]][start_pos[1]] = 0

        self.maze_view._reset_agent()

        observation = self._get_obs()
        info = self._get_info()

        self.cum_rew = 0
        self.steps_taken = 0
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
        reward = 0.0
        terminated = False
        truncated = False

        prev_pos = self._agent_location
        moved = self.maze_view.move_agent(BaseMazeEnv.ACTIONS[action])

        if moved:
            self._agent_location = np.array(self.maze_view._agent_position, dtype=np.int32)
            current_cell = tuple(self._agent_location)
            self.consecutive_invalid_moves = 0

            if current_cell not in self.visited_cell:
                self.non_visited[current_cell[0]][current_cell[1]] = 0
                if np.array_equal(self._agent_location, self._target_location):
                    reward = 1
                    terminated = True
                else:
                    new_dist = len(self.find_path(current_cell))
                    old_dist = len(self.find_path(tuple(prev_pos)))
                    
                    reward = (old_dist - new_dist) * 0.5 - 0.05
            else:
                reward -= 1 - math.exp(- 0.2 * (self.visited_cell.count(current_cell)))

            self.visited_cell.append(current_cell)

        else:
            self.consecutive_invalid_moves += 1
            reward -= 1 - math.exp(- 0.15 * (self.consecutive_invalid_moves))

        observation = self._get_obs()
        info = self._get_info()
        
        self.steps_taken+=1
        if self.steps_taken > self.max_steps_taken:
            truncated = True
            reward = -1
        self.cum_rew += reward
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
        Find the best next cell for the agent, considering path length and tie-breaking heuristics.
        
        Args:
            agent_pos (tuple): the position of the agent.
        
        Returns:
            tuple or None: The best next cell for the agent, or None if no valid move exists.
        """
        best_next_cell = agent_pos
        best_score = float("inf")  # Minimize path length + heuristic

        for dir in BaseMazeEnv.ACTIONS:
            
            next_pos = self.next_cell(agent_pos, dir)
            if not self.valid_cell(next_pos):
                continue

            # Compute path using A* with limited depth
            path = self.find_path(next_pos, max_depth= 2 * min(self.maze_shape[0],self.maze_shape[1]))

            if path:
                path_length = len(path)
                goal_pos = tuple(self._target_location)
                manhattan_dist = (abs(next_pos[0] - goal_pos[0])  + 
                                abs(next_pos[1] - goal_pos[1]))  # Manhattan distance

                score = path_length + 0.15 * manhattan_dist  # Weight Euclidean slightly to break ties

                if score < best_score:  
                    best_score = score
                    best_next_cell = next_pos

            # Early exit: If the next position is the goal, return immediately
            if next_pos == tuple(self._target_location):
                return next_pos
            
        return best_next_cell

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
    
class BaseConstantSizeEnv(BaseMazeEnv):
    """
    Template for a maze environment with constant size of the maze.
    """

    def __init__(self,maze_map,start_pos:tuple[int,int],goal_pos:tuple[int,int],maze_shape:tuple[int,int]):
        """
        Initialize the maze environment.
        Args:
            maze_map (array): the array representation of the maze where 0 represent walls, 1 the walkable tiles and 2 the goal.
            start_pos (tuple): the starting position of the agent.
            goal_pos (tuple): the goal position of the agent.
            maze_shape (tuple): the shape of the maze.
        """
        super(BaseConstantSizeEnv, self).__init__(maze_map,start_pos,goal_pos,maze_shape)

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(low = np.array([0,0]), high= np.array(self.maze_shape),dtype=int),
                "target": gym.spaces.Box(low = np.array([0,0]), high= np.array(self.maze_shape),shape=(2,),dtype=int),
                "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int)
            }
        )

class BaseVariableSizeEnv(BaseMazeEnv):
    """
    Template for a maze environment with variable size of the maze.
    """
    def __init__(self,maze_map,start_pos:tuple[int,int],goal_pos:tuple[int,int],maze_shape:tuple[int,int]):
        """
        Initialize the maze environment.
        Args:
            maze_map (array): the array representation of the maze where 0 represent walls, 1 the walkable tiles and 2 the goal.
            start_pos (tuple): the starting position of the agent.
            goal_pos (tuple): the goal position of the agent.
            maze_shape (tuple): the shape of the maze.
        """
        super(BaseVariableSizeEnv, self).__init__(maze_map,start_pos,goal_pos,maze_shape)

        self.observation_space = spaces.Dict(
            {
                "agent": gym.spaces.Box(low = np.array([0,0]), high= np.array(self.max_shape),dtype=int),
                "target": gym.spaces.Box(low = np.array([0,0]), high= np.array(self.max_shape),dtype=int),
                "best dir": gym.spaces.Box(-1,1,shape=(2,),dtype=int)
            }
        )

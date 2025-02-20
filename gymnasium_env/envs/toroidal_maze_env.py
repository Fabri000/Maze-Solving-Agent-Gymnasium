import numpy as np
from gymnasium_env.envs.base_maze_env import BaseMazeEnv
from lib.maze_generation import gen_maze_no_border
from lib.maze_view import ToroidalMazeView
from lib.a_star_algos.a_star_tor import astar_limited_partial


class ToroidalMazeEnv(BaseMazeEnv):
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

        start_pos, maze_map = gen_maze_no_border(maze_shape)
        goal_pos = [(r, c) for r in range(maze_shape[0]) for c in range(maze_shape[1]) if maze_map[r][c] == 2][-1]

        super(ToroidalMazeEnv, self).__init__(maze_map, start_pos, goal_pos,maze_shape)

        if render_mode == "human":
            self.maze_view = ToroidalMazeView(self.maze_map,self._start_pos,self._target_location,self.maze_shape)

        self.mazes.append([self._start_pos,self.maze_map])

        self.reset()

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
        return astar_limited_partial(self.maze_map,source,self._goal_pos,max_depth=max_depth)
    
    def update_visited_maze(self, remove: bool = True):
        """
        Update the game with a maze that was already learned at training time.
        Args:
            remove (bool): whether to remove the visited maze from the list of learned maze. Default: True.
        """
        self._start_pos, self.maze_map = self.mazes[self.next]

        if remove:
            self.mazes.remove([self._start_pos,self.maze_map])
        else:
            self.next+=1

        self._goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]    
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.maze_view.update_maze(self.maze_map,self._start_pos,self._goal_pos,self.maze_shape)
        self.reset()
    
    def update_maze(self):
        """
        Update the game maze.
        """
        self._start_pos , self.maze_map = gen_maze_no_border(self.maze_shape)

        self._goal_pos = [(r, c) for r in range(self.maze_shape[0]) for c in range(self.maze_shape[1]) if self.maze_map[r][c] == 2][0]
        self._target_location = np.array(self._goal_pos, dtype=np.int32)

        self.mazes.append([self._start_pos,self.maze_map])

        self.maze_view.update_maze(self.maze_map,self._start_pos,self._goal_pos,self.maze_shape)
        self.reset()
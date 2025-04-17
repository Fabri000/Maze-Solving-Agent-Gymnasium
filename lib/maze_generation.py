import random
import torch
from lib.a_star_algos.a_star import astar_limited_partial
from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation

def gen_maze(shape:tuple[int,int], algorithm:str="dfs"):
    """
    Create an array representation of a maze created using different types of algorithm.
    Args:
        shape (tuple): shape, in a tuple format, of the result maze.
        algorithm (str): algorithm to use for generation (currently supported "dfs" for deep first search, "r-prim" for random prim visit and "prim&kill" for prim&kill visit). Default: "dfs"
    Returns:
        tuple:
         - start_point (tuple): the starting position for the agent.
         - goal point (tuple): the goal position.
         - maze (array): the array representation of the maze where 0 represent walls, 1 the walkable tiles and 2 the goal.
    """
    rows,columns = (shape[0],shape[1])
    maze = [[0 for _ in range(rows)] for _ in range(columns)]
    
    start_point= (random.randrange(1, rows - 1, 2), random.randrange(1, columns - 1, 2))
    maze[start_point[0]][start_point[1]] = 1

    match algorithm:
        case "r-prim":
            random_prim_visit(maze,rows,columns,start_point)
        case "dfs":
            deept_first_visit(maze,rows,columns,start_point)
        case "prim&kill":
            prim_and_kill_visit(maze,rows,columns,start_point)
    
    goal_point = find_random_position(maze,1,start_point)
    maze[goal_point[0]][goal_point[1]] = 2

    return start_point , goal_point, maze

def gen_maze_no_border(shape:tuple[int,int], algorithm:str="dfs"):
    """
    Create an array representation, without borders, of a maze created using different types of algortithm.
    Args:
        shape (tuple): shape, in a tuple format, of the result maze.
        algorithm (str): algorithm to use for generation (currently supported "dfs" for deep first search and "r-prim" for random prim visit). Default: "dfs"
    Returns:
        tuple:
         - start_point (tuple): the starting position for the agent.
         - maze (array): the array representation of the maze where 0 represent walls, 1 the walkable tiles and 2 the goal.
    """
    extended_shape = (shape[0]+2,shape[1]+2)
    start_point, goal_point, maze = gen_maze(extended_shape,algorithm)

    difficulty = ComplexityEvaluation(maze,start_point,goal_point).difficulty_of_maze()

    maze = [row[1:len(row)-1] for row in maze[1:len(maze)-1]]
    start_point = (start_point[0]-1,start_point[1]-1)
    goal_point = (goal_point[0]-1,goal_point[1]-1)
    return start_point, goal_point, maze, difficulty


def random_prim_visit(maze, width: int, height: int, start_point: tuple[int, int]):
    """
    Implements a randomized Prim's algorithm for maze generation.

    Args:
        maze (list of lists): The maze grid, initially filled with walls (0).
        width (int): Width of the maze.
        height (int): Height of the maze.
        start_point (tuple): Starting position for the Prim's visit.
    """
    
    def get_neighbors(x, y):
        """Returns valid neighbors (inside the maze and within bounds)."""
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        return [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < height and 0 <= y + dy < width]

    # Initialize the maze: all walls (0)
    maze[start_point[0]][start_point[1]] = 1  # Mark starting point as a path
    frontier = set(get_neighbors(*start_point))  # Add neighboring cells to the frontier
    
    while frontier:
        fx, fy = random.choice(tuple(frontier))  # Pick a random frontier cell
        frontier.remove((fx, fy))  # Remove it from the frontier

        # Find neighbors that are already part of the maze
        neighbors = [(nx, ny) for nx, ny in get_neighbors(fx, fy) if maze[nx][ny] == 1]
        
        if neighbors:
            # Choose a random neighbor that is part of the maze
            nx, ny = random.choice(neighbors)

            # Remove the wall between the chosen frontier cell and its neighbor
            maze[fx][fy] = 1
            maze[(fx + nx) // 2][(fy + ny) // 2] = 1  # Remove the wall between the two cells

            # Add new valid frontier cells (neighbors of the current frontier cell)
            for new_nx, new_ny in get_neighbors(fx, fy):
                if maze[new_nx][new_ny] == 0:  # Only add unvisited cells to the frontier
                    frontier.add((new_nx, new_ny))

    return maze

def deept_first_visit(maze, width:int, height:int, start_point:tuple[int,int]):
    """
    Algorithm that implement a deep first visit for generating a maze.
    Args:
        maze (array): the array that will contain the final maze representation.
        width (int): width of the maze.
        height (int): height of the final maze.
        start_point (tuple): starting position for the prim visit.
    """
    stack = [start_point]
    
    while stack:
        x, y = stack[-1]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)
        found = False

        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 0 <= nx < height and 0 <= ny < width and maze[nx][ny] == 0:
                maze[x + dx][y + dy] = 1
                maze[nx][ny] = 1
                stack.append((nx, ny))
                found = True
                break 

        if not found:
            stack.pop()

def prim_and_kill_visit(maze,width:int,height:int, start_point:tuple[int,int]):
    """
    Algorithm that implement the Prim&Kill visit described into the paper "How to generate perfect mazes?" by Bellot et al.
    Args:
        maze (array): the array that will contain the final maze representation.
        width (int): width of the maze.
        height (int): height of the final maze.
        start_point (tuple): starting position for the prim visit.
    """
    size = (width,height)
    unmarked = set([(i,j) for i in range(1,width,2) for j in range(1,height,2) if i%2 !=0 and i%2!=0])
    for i,j in unmarked:
        maze[i][j]=1
    marked = set([start_point])
    unmarked.discard(start_point)

    current = start_point
    maze[current[0]][current[1]]=1
    unmarked,marked = random_walk(maze,unmarked,marked,current,size)

    while len(unmarked)!=0:
        current = random.choice(tuple([p for p in marked if len(set([(p[0] + dx, p[1] + dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)] if 0 <= p[0] + dx < height and 0 <= p[1] + dy < width]).intersection(unmarked)) != 0]))
        unmarked,marked = random_walk(maze,unmarked,marked,current,size)

def random_walk(maze,unmarked,marked,current,size):
    """
    Do a random walk on the maze starting from current.
    Args:
        maze (array): the array representing the current maze
        unmarked (array): contains the position not visited.
        marked (array): contains the position already visited.
        current (tuple[int,int]): the current position.
        size (tuple[int,int]):size of the maze.
    """
    
    width,height = size
    neighbors = set([(current[0] + dx, current[1] + dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)] if 0 <= current[0] + dx < height and 0 <= current[1] + dy < width])
    inters = unmarked.intersection(neighbors) 
    
    while len(inters)!=0:

        maze[current[0]][current[1]]=1

        next = random.choice(tuple(inters))
        wx,wy = (current[0] + ((next[0]-current[0]) // 2) , current[1]+((next[1]-current[1]) // 2))
        maze[wx][wy]=1

        current = next 
        neighbors = set([(current[0] + dx, current[1] + dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)] if 0 <= current[0] + dx < height and 0 <= current[1] + dy < width])
        
        unmarked.remove(current)
        marked.add(current)
        
        inters = unmarked.intersection(neighbors)
    
    return  unmarked,marked

def find_random_position(maze, val:int, start_point:tuple[int,int]):
    """
    Find a random position into the maze associated to a specific value
    Args:
        maze (array): maze representation.
        val (int): the value to fine.
        start_point (tuple): the starting point used to generate the maze.
    Returns:
        chosed (tuple): the position found with the value.
    """
    positions = [(r, c) for r in range(1,len(maze),2) for c in range(1,len(maze[0]),2) if maze[r][c] == val]
    positions.remove(start_point)

    candidates = []
    for position in positions:
        i,j = position
        neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if maze[i + dx][j + dy] != 0)
        if neighbors == 1:
            candidates.append(position)

    if not candidates:
        return None
    
    chosed = candidates[0]
    max_dist = len(astar_limited_partial(maze,start_point,chosed))
    for candidate in candidates:
        dist_candidate = len(astar_limited_partial(maze,start_point,candidate))
        if dist_candidate > max_dist:
            chosed = candidate
            max_dist = dist_candidate

    return chosed

def generate_collection_of_mazes(shape:tuple[int,int],num_mazes:int,algorithms:list[str]=["dfs","r-prim","prim&kill"]):
    """
    Generate a set of mazes
    Args:
        shape (tuple): size of the mazes to generate.
        num_mazes (int): the dimension of the final set.
    Returns:
        maze_set (array): set of generated mazes.
    """
    maze_set = []
    while len(maze_set) < num_mazes:

        start_point,_, maze = gen_maze(shape, random.choice(algorithms))
        
        maze_tensor = torch.tensor(maze)
        
        
        tile_mask = (maze_tensor == 1).int() # tiles are represented by a 1 in the matrix representation)
        wall_mask = (maze_tensor == 0).int() # walls are represented by a 0 in the matrix representation)
        non_visited = (maze_tensor != 0).int()
        non_visited[start_point[0]][start_point[1]] = 0
        
        final_tensor = torch.stack([wall_mask,tile_mask,non_visited])

        if not any(torch.equal(final_tensor, maze) for maze in maze_set):
            maze_set.append(final_tensor)

    return maze_set

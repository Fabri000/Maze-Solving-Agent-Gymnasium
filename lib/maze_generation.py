import random
import torch

def gen_maze(shape:tuple[int,int], algorithm:str="dfs"):
    """
    Create an array representation of a maze created using different types of algortithm.
    Args:
        shape (tuple): shape, in a tuple format, of the result maze.
        algorithm (str): algorithm to use for generation (currently supported "dfs" for deep first search and "r-prim" for random prim visit). Default: "dfs"
    Returns:
        tuple:
         - start_point (tuple): the starting position for the agent.
         - maze (array): the array representation of the maze where 0 represent walls, 1 the walkable tiles and 2 the goal.
    """
    rows,columns = shape
    maze = [[0 for _ in range(rows)] for _ in range(columns)]
    
    start_point= (random.randrange(1, rows - 1, 2), random.randrange(1, columns - 1, 2))
    maze[start_point[0]][start_point[1]] = 1

    match algorithm:
        case "r-prim":
            random_prim_visit(maze,rows,columns,start_point)
        case "dfs":
            deept_first_visit(maze,rows,columns,start_point)

    win_pos = find_random_position(maze,1,start_point)
    maze[win_pos[0]][win_pos[1]] = 2

    return start_point , maze

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
    start_point, maze = gen_maze(extended_shape,algorithm)

    maze = [row[1:len(row)] for row in maze[1:len(maze)]]
    start_point = (start_point[0]-1,start_point[1]-1)
    return start_point , maze

def random_prim_visit(maze,width:int,height:int,start_point:tuple[int,int]):
    """
    Algorithm that implement a random prim visit for generating a maze.
    Args:
        maze (array): the array that will contain the final maze representation.
        width (int): width of the maze.
        height (int): height of the final maze.
        start_point (tuple): starting position for the prim visit.
    """
    frontier = []
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        nx, ny = start_point[0] + dx, start_point[1] + dy
        if 0 < nx < height - 1 and 0 < ny < width - 1:  # Avoid exit from the border
            frontier.append((nx, ny))
    
    while frontier:
        # randomizattion of next cell
        fx, fy = random.choice(frontier)
        frontier.remove((fx, fy))
        
        # find neighbor cell already explored
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < height and 0 <= ny < width and maze[nx][ny] == 1:
                neighbors.append((nx, ny))
        
        if neighbors:
            # randoom neighbor
            nx, ny = random.choice(neighbors)
            
            # Connect cells into frontier to the neighbor
            maze[fx][fy] = 1
            maze[(fx + nx) // 2][(fy + ny) // 2] = 1  # remove wall
            
            # Add cell to frontier
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = fx + dx, fy + dy
                if 0 < nx < height - 1 and 0 < ny < width - 1 and maze[nx][ny] == 0:
                    frontier.append((nx, ny))



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
        x, y = stack.pop()
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 0 < nx < height - 1 and 0 < ny < width - 1 and maze[nx][ny] == 0:
                maze[x + dx][y + dy] = 1
                maze[nx][ny] = 1
                stack.append((nx, ny))


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
    positions = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == val]
    positions.remove(start_point)

    if not positions:
        return None
    
    chosed = random.choice(positions)
    while abs(chosed[0]-start_point[0])+abs(chosed[1]-start_point[1]) < max(len(maze),len(maze[0])) - max(len(maze),len(maze[0]))/2 and (chosed[0] == 0 or chosed[1] == 0 or chosed[0] == len(maze)-1 or chosed[1] == len(maze[0])-1):
        chosed = random.choice(positions)

    return chosed

def generate_collection_of_mazes(shape:tuple[int,int],num_mazes:int):
    """
    Generate a set of mazes
    Args:
        shape (tuple): size of the mazes to generate.
        num_mazes (int): the dimension of the final set.
    Returns:
        maze_set (array): set of generated mazes.
    """
    maze_set = []
    algos = ["dfs","r-prim"]
    while len(maze_set) < num_mazes:

        _, maze = gen_maze(shape, random.choice(algos))
        
        maze_tensor = torch.tensor(maze)
        
        goal_mask = (maze_tensor == 2).int() # goal is represented by a 2 in the matrix representation
        tile_mask = (maze_tensor == 1).int() # tiles are represented by a 1 in the matrix representation)
        wall_mask = (maze_tensor == 0).int() # walls are represented by a 0 in the matrix representation)
        
        final_tensor = torch.stack([wall_mask,tile_mask,goal_mask])

        if not any(torch.equal(final_tensor, maze) for maze in maze_set):
            maze_set.append(final_tensor)

    return maze_set

import torch
import numpy as np

def extract_submaze(maze,non_visited,position:tuple[int,int],shape:int):
    '''
        extract a square window of a maze of a specified shape given an anchor position.
        Args:
            maze (array): matrix representation of the maze.
            position (tuple): the anchor position.
            shape (int): shape of window.
        Returns:
            window
    '''
    maze_shape = len(maze)

    k = shape // 2

    row_start=row_end=col_start=col_end= -1
    
    if shape  == maze_shape:
        return maze,non_visited,position
    
    # Row boundary checks
    if position[0] - k >= 0 and position[0] + k < maze_shape:
        row_start = position[0] - k
        row_end = position[0] + k + 1  # +1 to include the end index
    elif position[0] - k < 0 and position[0] + k < maze_shape:
        row_start = 0
        row_end = shape  # size of the window
    elif position[0] - k >= 0 and position[0] + k >= maze_shape:
        row_start = maze_shape - shape  # ensure sub-region fits within bounds
        row_end = maze_shape

    # Column boundary checks
    if position[1] - k >= 0 and position[1] + k < maze_shape:
        col_start = position[1] - k
        col_end = position[1] + k + 1  # +1 to include the end index
    elif position[1] - k < 0 and position[1] + k < maze_shape:
        col_start = 0
        col_end = shape  # size of the window
    elif position[1] - k >= 0 and position[1] + k >= maze_shape:
        col_start = maze_shape - shape
        col_end = maze_shape

    ris = [row[col_start: col_end] for row in maze[row_start: row_end]]

    # Compute the corresponding position in the original maze
    player_position = (
        position[0] - row_start,
        position[1] - col_start
    )
    non_visited = [row[col_start: col_end] for row in non_visited[row_start: row_end]]

    return ris ,non_visited, player_position

def extract_submaze_toroid(maze,non_visited, position: tuple[int, int], shape: int):
    '''
    Extract a square window of a maze with toroidal topology.
    Args:
        maze (array): matrix representation of the maze.
        position (tuple): the anchor position.
        shape (int): shape of the window.
    Returns:
        window (array): extracted submaze.
    '''
    maze_shape = len(maze)
    k = shape // 2
    
    if shape == maze_shape:
        return maze, position
    
    rows = [(position[0] + i - k) % maze_shape for i in range(shape)]
    cols = [(position[1] + i - k) % maze_shape for i in range(shape)]
    
    submaze = np.array([[maze[r][c] for c in cols] for r in rows])
    player_position = (k, k)

    non_visited = np.array([[non_visited[r][c] for c in cols] for r in rows])

    return submaze,non_visited,player_position

def get_mask_tensor(maze,non_visited,position):
    '''
    Generate a tensor containing the mask for the given maze.
    Args:
        maze
        position
    Returns:
        torch.tensor
    '''
    maze_tensor = torch.tensor(maze)
        
    cell_mask = (maze_tensor == 1).int() # mask for the walkable cells
    wall_mask = (maze_tensor == 0).int() # walls are represented by a 0 in the matrix representation)
    visited_mask = torch.tensor(non_visited).int() # visited cells are represented by a 1 in the matrix representation
    
    final_tensor = torch.stack([wall_mask,cell_mask,visited_mask])
    
    return final_tensor.float()

def get_decision_mask(maze_tensor):
    H, W = maze_tensor.shape
    decision_mask = torch.zeros_like(maze_tensor)

    for i in range(1,H,2):
        for j in range(1,W,2):
            if maze_tensor[i, j] != 1:
                continue

            neighbors = 0
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if maze_tensor[ni, nj] == 1:
                        neighbors += 1

            if neighbors >= 3:
                decision_mask[i, j] = 1

    return decision_mask

def get_direction_mask(maze,position):
    '''
    Generate a tensor containing the direction mask for the given maze.
    Args:
        maze
        position
    Returns:
        torch.tensor
    '''
    mask = np.ones(4, dtype=np.int32)
    i,j = position

    k = 0
    for di, dj in [(1,0),(-1,0), (0,1), (0,-1)]:
        ni, nj = i + di, j + dj
        if maze[ni][nj] == 0:
            mask[k] = 0
        k+=1

    return mask

def get_toroidal_direction_mask(maze,position):
    '''
    Generate a tensor containing the direction mask for the given maze.
    Args:
        maze
        position
    Returns:
        torch.tensor
    '''
    mask = np.ones(4, dtype=np.int32)
    i,j = position

    k = 0
    for di, dj in [(1,0),(-1,0), (0,1), (0,-1)]:
        ni, nj = (i + di)% len(maze) , (j + dj)% len(maze[0])
        if maze[ni][nj] == 0:
            mask[k] = 0
        k+=1

    return mask
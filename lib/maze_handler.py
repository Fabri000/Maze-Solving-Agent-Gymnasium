import torch

def extract_submaze(maze,position:tuple[int,int],shape:int):
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
    return ris 

def get_mask_tensor(maze):
    '''
    Generate a tensor containing the mask for the given maze.

    Args:
        maze
    Returns:
        torch.tensor
    '''

    maze_tensor = torch.tensor(maze)
        
    goal_mask = (maze_tensor == 2).int() # goal is represented by a 2 in the matrix representation
    tile_mask = (maze_tensor == 1).int() # tiles are represented by a 1 in the matrix representation)
    wall_mask = (maze_tensor == 0).int() # walls are represented by a 0 in the matrix representation)
    
    final_tensor = torch.stack([wall_mask,tile_mask,goal_mask])
    
    return final_tensor.float()
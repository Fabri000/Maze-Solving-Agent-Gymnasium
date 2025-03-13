from lib.maze_generation import gen_maze
import numpy as np

import random

def generate_maze_no_outer_walls(rows, cols):
    # Initialize all cells as walls (0)
    maze = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Start at the top-left corner (0, 0)
    start_row, start_col = 1,1
    maze[start_row][start_col] = 1  # Mark start as path
    
    stack = [(start_row, start_col)]
    
    # Directions: up, down, left, right (each moves 1 step)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        current_row, current_col = stack.pop()
        
        # Collect unvisited neighbors (1 step away)
        neighbors = []
        for dr, dc in directions:
            nr = current_row + dr
            nc = current_col + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                # Count how many neighbors are already connected to the path
                connected = 0
                for ddr, ddc in directions:
                    nnr = nr + ddr
                    nnc = nc + ddc
                    if 0 <= nnr < rows and 0 <= nnc < cols and maze[nnr][nnc] == 1:
                        connected += 1
                if connected <= 1:  # Avoid creating cycles
                    neighbors.append((nr, nc))
        
        if neighbors:
            # Randomize neighbors to choose a random direction
            random.shuffle(neighbors)
            next_row, next_col = neighbors[0]
            maze[next_row][next_col] = 1  # Carve path
            
            # Push current and next cells to the stack
            stack.append((current_row, current_col))
            stack.append((next_row, next_col))
    
    return maze

# Example usage (5x5 maze with no outer walls):
maze = generate_maze_no_outer_walls(7, 7)
for row in maze:
    print(row)

maze = np.array(maze)
maze = maze[1::2, 1::2]
print(maze)
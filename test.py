from lib.maze_generation import gen_maze


start_pos, goal_point, maze = gen_maze((15,15),"prim&kill")
print(maze)
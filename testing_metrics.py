from lib.maze_generation import gen_maze
from lib.maze_difficulty_evaluation.metrics_calculator import MetricsCalulator
from lib.a_star_algos.a_star import astar_limited_partial

"""maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 2, 1, 1, 0, 1, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
sol_path = [(9, 9), (10, 9), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (10, 13), (9, 13), (8, 13), (7, 13), (6, 13), (5, 13), (4, 13), (3, 13), (3, 12), (3, 11), (2, 11), (1, 11), (1, 10), (1, 9), (1, 8), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (5, 6), (5, 5), (6, 5), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (12, 5), (13, 5)]
goal_pos = (13, 5)
start_pos = (9, 9)
maze_size = (15,15)"""


"""maze =[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

start_pos = (1,1)
maze_shape = (15,15)
goal_pos = goal_pos = [(r, c) for r in range(maze_shape[0]) for c in range(maze_shape[1]) if maze[r][c] == 2][-1]

sol_path = astar_limited_partial(maze,start_pos,goal_pos)"""

start_pos,maze = gen_maze((15,15),"r-prim")

maze_shape = (len(maze),len(maze[0]))
goal_pos = goal_pos = [(r, c) for r in range(maze_shape[0]) for c in range(maze_shape[1]) if maze[r][c] == 2][-1]

sol_path = astar_limited_partial(maze,start_pos,goal_pos)

metrics_calculator = MetricsCalulator(maze, len(sol_path))

L_s = metrics_calculator.calculate_L(sol_path)


print("L_s ", L_s)
print("Sum L_d = 1 - L_s ", (1-L_s))
print("Sum L_d = Sum L_fd + Sum L_bd ", (metrics_calculator.calculate_L_sharp(sol_path,"BDE")+ metrics_calculator.calculate_L_sharp(sol_path,"FDE")))

print("DE = AC+BDE+FDE ", metrics_calculator.calculate_DE(sol_path))
print("DE = J_s +2*CR_s ", metrics_calculator.calculate_J(sol_path)+ 2*metrics_calculator.calculate_CR(sol_path))

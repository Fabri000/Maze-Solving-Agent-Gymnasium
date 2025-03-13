from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation
from lib.maze_generation import gen_maze

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
goal_pos = [(r, c) for r in range(maze_shape[0]) for c in range(maze_shape[1]) if maze[r][c] == 2][-1]

c_e = ComplexityEvaluation(maze,start_pos,goal_pos)

print("difficulty ", c_e.difficulty_of_maze())
print("complexity ", c_e.complexity_of_maze())"""


difficulties = []
complexities = []


for i in range(20):
    start_pos,maze = gen_maze((41,41))
    maze_shape = (len(maze),len(maze[0]))
    goal_pos = goal_pos = [(r, c) for r in range(maze_shape[0]) for c in range(maze_shape[1]) if maze[r][c] == 2][-1]

    c_e = ComplexityEvaluation(maze,start_pos,goal_pos)

    difficulties.append(c_e.difficulty_of_maze())
    complexities.append(c_e.complexity_of_maze())

print("mean difficuly ", sum(difficulties)/len(difficulties))
print("max difficulty ", max(difficulties) )
print("mean complexity", sum(complexities)/len(complexities))


from datetime import datetime
import logging
from tqdm import tqdm
from numpy import mean, median
from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation
from lib.maze_difficulty_evaluation.metrics_calculator import MetricsCalulator
from lib.a_star_algos.a_star import astar_limited_partial
from lib.maze_generation import gen_maze


def init_logger(log_name:str):

    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(filename=f"metrics_{file_name}.log",filemode="a",format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level= logging.DEBUG)
    logger = logging.getLogger(log_name)
    return logger

MAZE_SIZE=(81,81) #must be doubled to reconnect to the thin maze representation
NUM_SAMPLES = 500
AlGORITHMS = ["dfs","r-prim", "prim&kill"]


results = {}
for algo in tqdm(AlGORITHMS):
    difficulties = []
    complexities = []
    path_lenghts = []
    dead_ends = []
    decision_points = []
    for _ in tqdm(range(NUM_SAMPLES)):
        start_point,goal_point,maze = gen_maze(MAZE_SIZE,algo)
        c_e = ComplexityEvaluation(maze,start_point,goal_point)
        difficulties.append(c_e.difficulty_of_maze())
        complexities.append(c_e.complexity_of_maze())
        solution = astar_limited_partial(maze,start_point,goal_point)
        m_c = MetricsCalulator(maze,len(solution))
        path_lenghts.append(m_c.calculate_L(solution))
        dead_ends.append(m_c.calculate_DE(solution))
        decision_points.append(m_c.calculate_D(solution))


    results[algo] = {" McCledon difficulty": mean(difficulties), "Max Difficulty":max(difficulties), "McCledon complexity":mean(complexities),"L":mean(path_lenghts),"DE":mean(dead_ends),"D":mean(decision_points)}

logger = init_logger("Maze_generation_metrics")
logger.info(f"Results of the maze generation metrics evaluation on {NUM_SAMPLES} samples of size {MAZE_SIZE}")
for algo in results:
    logger.info(f"Algorithm {algo} \n {results[algo]}")

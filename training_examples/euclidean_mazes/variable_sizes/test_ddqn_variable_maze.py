import sys
import os

from tqdm import tqdm

# Get the absolute path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Add the root directory to sys.path
sys.path.append(root_dir)

import gymnasium as gym
import torch
import torch_directml

from lib.maze_difficulty_evaluation.maze_complexity_evaluation import ComplexityEvaluation

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from gymnasium_env.envs.simple_variable_maze_env import SimpleEnrichVariableMazeEnv
from lib.trainers.off_policy_trainer import NeuralOffPolicyTrainer
from agents.ddqn_agent import DDQNAgent
from lib.logger_inizializer import init_logger

import torch_directml


maze_max_shape = (81,81)
n_episodes = 200
learning_rate=1e-3
starting_epsilon=0.95
final_epsilon=0.1
epsilon_decay= eps_dec = ((maze_max_shape[0]-1)*(maze_max_shape[1]-1) // 2)
discount_factor=0.7
eta = 1e-2
batch_size=128

device = torch_directml.device()

env = SimpleEnrichVariableMazeEnv(max_shape=maze_max_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = DDQNAgent(env,learning_rate=learning_rate,starting_epsilon=starting_epsilon,final_epsilon=final_epsilon,epsilon_decay=epsilon_decay,discount_factor=discount_factor,eta=eta,batch_size=batch_size,memory_size=50000,target_update_frequency=1,device=device)

logger = init_logger("Agent_log","logs/variable_ddqn_logs")
logger.info(f"Training starting on variable mazes with dimension variable between {SimpleEnrichVariableMazeEnv.START_SHAPE} and {maze_max_shape}")
logger.debug(f"Hyperparameters: lr {learning_rate} | eps_init {starting_epsilon} | eps_end {final_epsilon} | eps_dec {epsilon_decay} | discount_fact {discount_factor} | batch size {batch_size}")

c_e = ComplexityEvaluation(env.env.maze_map,env.env._start_pos,tuple(env.env._target_location))
logger.debug(f'Learning new maze| maze of shape {env.env.maze_shape} | maze difficulty {c_e.difficulty_of_maze()}')

trainer = NeuralOffPolicyTrainer(agent,env,device,logger)

trainer.train(n_episodes)

logger.info("Checking if the agent remember how to solve maze already seen")
trainer.test(len(env.env.mazes),new = False)

logger.info(f'Start testing on new mazes')
trainer.test(150, new = True)

logger.info(f'Infer on different sizes in training range')
for dim in tqdm(range(SimpleEnrichVariableMazeEnv.START_SHAPE[0],maze_max_shape[0],12)):
    trainer.infer(20,"r-prim",(dim,dim))

logger.info(f'Infer on different sizes not in training range')
for dim in tqdm([83,95,107,119,131]):
    trainer.infer(20,"r-prim",(dim,dim))

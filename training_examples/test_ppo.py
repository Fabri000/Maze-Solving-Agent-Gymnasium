import os
import sys
import gymnasium as gym
import torch
import torch_directml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from gymnasium_env.envs.simple_maze_env import SimpleEnrichMazeEnv
from agents.ppo_agent import PPOAgent
from lib.logger_inizializer import init_logger
from lib.trainers.ppo_trainer import PPOTrainer

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)


maze_shape = (17,17)

n_episodes = 1000
update_freq = 5
actor_lr= 3e-4
critic_lr= 1e-4
gamma=0.9
batch_size = 64
ppo_steps= 8

env = SimpleEnrichMazeEnv(maze_shape)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

device = torch_directml.device()

agent = PPOAgent(actor_lr,critic_lr,gamma,batch_size,ppo_steps,env,device)

log_dir = "logs/ppo_agent"
logger = init_logger("Agent_log",log_dir)

logger.info(f"Maze of shape {env.env.get_maze_shape()} | total episodes of training {n_episodes}")
logger.debug(f"Hyperparameter of training: actor learning rate {actor_lr}| critic learning rate {critic_lr}| update frequency {update_freq} | gamma {gamma}| batch size {batch_size} | ppo steps {ppo_steps}")

trainer = PPOTrainer(env,agent,logger,device)

trainer.train(n_episodes,update_freq)

logger.info("Checking if the agent remember how to solve maze already seen")
trainer.test(len(env.env.mazes),new = False)
logger.info(f'Start testing on new mazes')
trainer.test(75, new = True)
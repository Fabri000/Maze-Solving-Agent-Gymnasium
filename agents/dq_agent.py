import numpy as np
import math
from collections import defaultdict

class DQAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        gamma: float,
        eta:float): #discount factor
 
        self.env = env
        self.q_a_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.q_b_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.learning_rate = learning_rate

        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.epsilon = initial_epsilon

        self.gamma = gamma
        self.eta = eta

        self.steps_done = 0

        self.training_error = []

    def get_action(self,obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        epsilon_threshold = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if np.random.random() < epsilon_threshold:
            return self.env.action_space.sample()
        else:
            q_table = self.q_a_values[str(obs)]+self.q_a_values[str(obs)]
            return int(np.argmax(q_table))
    
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        td_error = 0
        if np.random.random()<0.5: #update Q_A
            best_action= self.get_action(next_obs)
            td_error = reward + self.gamma * self.q_b_values[str(next_obs)][best_action] - self.q_a_values[str(obs)][action]
            self.q_a_values[str(obs)][action] = self.q_a_values[str(obs)][action] + self.learning_rate *  td_error
        else: #update Q_B
            best_action= self.get_action(next_obs)
            td_error = reward + self.gamma * self.q_a_values[str(next_obs)][best_action] - self.q_b_values[str(obs)][action]
            self.q_b_values[str(obs)][action] = self.q_b_values[str(obs)][action] + self.learning_rate * td_error

        self.training_error.append(td_error)

    def update_steps_done(self):
        self.steps_done = self.steps_done // 4

    def update_hyperparameter(self,is_better:bool):
        if is_better:
            self.gamma = self.gamma + self.eta
        else:
            self.gamma = self.gamma - self.eta

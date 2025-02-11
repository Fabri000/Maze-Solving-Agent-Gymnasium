from __future__ import annotations

from collections import defaultdict

import numpy as np
import math

class QAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.initial_epsilon=initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.steps_done = 0

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        epsilon_threshold = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if np.random.random() < epsilon_threshold:
            return self.env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[str(obs)]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[str(next_obs)])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[str(obs)][action]
        )
        self.q_values[str(obs)][action] = (
            self.q_values[str(obs)][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    
    def update_epsilon_decay(self,maze_shape, n_episodes:int):
        self.epsilon_decay = 0.25 * maze_shape[0] * maze_shape[0] * n_episodes
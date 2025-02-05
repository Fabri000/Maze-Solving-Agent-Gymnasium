import random
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class PrioritizedReplayMemory(object):

    def __init__(self,capacity:int,alpha:float,epsilon:float):
        self.memory = deque([],maxlen=capacity)
        self.priorities = deque([],maxlen=capacity)
        self.alpha = alpha
        self.epsilon = epsilon

    def push(self,*args):
        self.memory.append(Transition(*args))
        max_priority = max(self.priorities, default=1.0)
        self.priorities.append(max_priority)

    def sample(self,batch_size, beta = 0.4):
        priorities = np.array(self.priorities,dtype= np.float32)
        p = priorities ** self.alpha
        p /= np.sum(p)

        indexs = np.random.choice(len(self.memory),batch_size,p=p)
        samples = [self.memory[index] for index in indexs]


        total = len(self.memory)
        weights = (total * p[indexs]) ** (-beta)
        weights /= weights.max()

        return samples,indexs,np.array(weights, dtype=np.float32)
    
    def update_priorities(self,indexs,deltas):
        for idx, delta in zip(indexs,deltas):
            self.priorities[idx] = delta+ self.epsilon

    def __len__(self):
        return len(self.memory)

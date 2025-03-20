import random
import torch
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
   
    def __init__(self, capacity):
        
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class SequentialExperienceReplayMemory(object):

    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
        self.buffer = []

    def add(self,done,*args):
        self.buffer.append(Transition(*args))
        if done:
            to_add = self.buffer.copy()
            self.buffer = []
            self.memory.append(to_add)

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
            
    def __len__(self):
        return len(self.memory)

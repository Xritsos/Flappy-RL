"""Implement the Replay Buffer based on:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random
from collections import namedtuple, deque

# Transition that maps (state, action) pairs to their (next_state, reward) result
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory:
    """A cyclic buffer of bounded size that holds the transitions observed recently"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Selects a random batch of transitions for training."""
        random.seed(22)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
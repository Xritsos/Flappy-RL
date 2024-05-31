"""Implement the Replay Buffer."""

import random
from collections import namedtuple


class ReplayMemory:
    """A cyclic buffer of bounded size that holds the transitions observed recently"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
        # Transition that maps (state, action) pairs to their (next_state, reward) result
        self.Transition = namedtuple('Transition', 
                                     ('state', 'action', 'next_state', 'reward', 'terminal'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Selects a random batch of transitions for training."""
        random.seed(22)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
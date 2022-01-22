import random
from collections import namedtuple

from ngu.models.common.type import Sequence


class ReplayMemory:

    def __init__(self, replay_capacity):
        self.memory = []
        self.next_idx = 0
        self.replay_capacity = replay_capacity

    def insert(self, sequence: Sequence):
        self.memory[self.next_idx] = sequence
        self.next_idx = (self.next_idx + 1) % self.replay_capacity

    def sample(self, batch_size):
        idxs = random.choice(range(self.replay_capacity), batch_size)
        return idxs

    def __len__(self):
        return len(self.memory)

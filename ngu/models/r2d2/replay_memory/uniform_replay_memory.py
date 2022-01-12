import random
from collections import deque


from ngu.models.r2d2.replay_memory import ReplayMemory


class UniformReplayMemory(ReplayMemory):
    """Uniform replay memory without prioritization"""

    def __init__(self, replay_capacity):
        self.replay_capacity = replay_capacity
        self.memory = deque(maxlen=self.replay_capacity)

    def insert(self, transitions):
        # TODO(minho) Sequence?
        self.memory.extend(transitions)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

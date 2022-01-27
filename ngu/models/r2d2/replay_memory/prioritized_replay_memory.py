from collections import deque

from ngu.models.r2d2.replay_memory.sumtree import SumTree


class PrioritizedReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.sequences = deque()
        self.priorities = SumTree()

    def push(self, sequences, priorities):
        self.sequences.extend(sequences)
        self.priorities.extend(priorities)

    def sample(self, batch_size):
        idxs, prios = self.priorities.prioritized_sample(batch_size)
        return [self.sequences[i] for i in idxs], prios, idxs

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def remove_to_fit(self):
        if len(self.priorities) - self.capacity <= 0:
            return
        for _ in range(len(self.priorities) - self.capacity):
            self.sequences.popleft()
            self.priorities.popleft()

    def __len__(self):
        return len(self.sequences)

    @property
    def total_prios(self):
        return self.priorities.root.value

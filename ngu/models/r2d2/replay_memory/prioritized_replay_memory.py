import random
from collections import deque

from baselines.common.segment_tree import SegmentTree, SumSegmentTree, MinSegmentTree

from ngu.models.r2d2.replay_memory import ReplayMemory, Sequence, Transition


# TODO(minho): Implement PrioritizedReplayMemory.
class PrioritizedReplayMemory(ReplayMemory):

    def __init__(self, replay_capacity, replay_priority_exponent):
        """Create replay memory prioritized by TD error

        Args:
            replay_capacity: Capacity of the memory.
            replay_priority_exponent: How much prioritization is used (0: uniform sample, 1: full prioritization)
        """
        super(PrioritizedReplayMemory, self).__init__(replay_capacity)
        self.rpl_exp = replay_priority_exponent
        # Segment tree size must be power of two.
        seg_cap = 1
        while seg_cap < self.replay_capacity:
            seg_cap *= 2

        self.sum_tree = SumSegmentTree(seg_cap)
        self.min_tree = MinSegmentTree(seg_cap)
        self._max_priority = 1.0

    def insert(self, sequence: Sequence):
        curr_idx = self.next_idx
        super().insert(sequence) # Calling this will modify next_idx, thus save in curr_idx before.
        self.sum_tree[curr_idx] = self._max_priority ** self.rpl_exp
        self.min_tree[curr_idx] = self._max_priority ** self.rpl_exp

    def sample(self, batch_size):
        idxes = self._sample_propotional(batch_size)
        return idxes  # TODO(minho): Importance weighting needed?

    def _sample_propotional(self, batch_size):
        res = []
        p_total = self.sum_tree.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self.sum_tree.find_prefixsum_idx(mass)
            res.append(idx)

        return res

    def update_priorities(self, idxs, new_priorities):
        """
        Args:
            idxs: Batch idexs of sampled sequences.
            new_priorities: Newly calculated priorities to put at idxs.
        """
        for i, p in zip(idxs, new_priorities):
            self.sum_tree[i] = p**self.rpl_exp
            self.min_tree[i] = p**self.rpl_exp
            self._max_priority = max(self._max_priority, p)

import torch

from ngu.models.r2d2.replay_memory.sumtree import SumTree


class PrioritizedReplayMemory:

    def __init__(self, capacity, seq_len, hidden_units):
        self.capacity = capacity
        self.seq_len = seq_len

        # Memory elements.
        self.state = torch.zeros((capacity, seq_len, 1, 84, 84), dtype=torch.float32)
        self.prev_action = torch.zeros((capacity, seq_len, 1), dtype=torch.int64)
        self.curr_action = torch.zeros((capacity, seq_len, 1), dtype=torch.int64)
        self.intrinsic_reward = torch.zeros((capacity, seq_len, 1), dtype=torch.float32)
        self.extrinsic_reward = torch.zeros((capacity, seq_len, 1), dtype=torch.float32)
        self.augmented_reward = torch.zeros((capacity, seq_len, 1), dtype=torch.float32)
        self.nstep_reward = torch.zeros((capacity, seq_len, 1), dtype=torch.float32)

        self.init_hx = torch.zeros((capacity, hidden_units), dtype=torch.float32)
        self.init_cx = torch.zeros((capacity, hidden_units), dtype=torch.float32)
        self.intrinsic_factor = torch.zeros((capacity, 1), dtype=torch.float32)
        self.discount_factor = torch.zeros((capacity, 1), dtype=torch.float32)

        self.priorities = SumTree()

        self.mem_idx = 0
        self.mem_pushed = 0

    def push(self, state, prev_act, curr_act, intr_rew, extr_rew, agmt_rew, nstep_rew, init_hx, init_cx,
             intr_factor, disc_factor, priorities):
        """Push batch of states and others into prioritized replay memory."""
        for i in range(len(state)):
            self.state[self.mem_idx].copy_(state[i])
            self.prev_action[self.mem_idx].copy_(prev_act[i])
            self.curr_action[self.mem_idx].copy_(curr_act[i])
            self.intrinsic_reward[self.mem_idx].copy_(intr_rew[i])
            self.extrinsic_reward[self.mem_idx].copy_(extr_rew[i])
            self.augmented_reward[self.mem_idx].copy_(agmt_rew[i])
            self.nstep_reward[self.mem_idx].copy_(nstep_rew[i])
            self.init_hx[self.mem_idx].copy_(init_hx[i])
            self.init_cx[self.mem_idx].copy_(init_cx[i])
            self.intrinsic_factor[self.mem_idx].copy_(intr_factor[i])
            self.discount_factor[self.mem_idx].copy_(disc_factor[i])

            self.priorities.append(priorities[i])
            if len(self.priorities) >= self.capacity:
                self.priorities.popleft()

            self.mem_idx = (self.mem_idx + 1) % self.capacity
            self.mem_pushed += 1

    def sample(self, batch_size):
        idxs, prios = self.priorities.prioritized_sample(batch_size)
        return ((self.state[idxs, ...], self.prev_action[idxs, ...], self.curr_action[idxs, ...],
                 self.intrinsic_reward[idxs, ...], self.extrinsic_reward[idxs, ...],
                 self.augmented_reward[idxs, ...], self.init_recurr_state[idxs, ...],
                 self.intrinsic_factor[idxs, ...], self.discount_factor[idxs, ...]), prios, idxs)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """In placed of real memory size (which is fixed size), use mem_pushed count to get the size of memory size."""
        return self.mem_pushed

    @property
    def total_prios(self):
        return self.priorities.root.value

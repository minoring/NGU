from abc import ABC, abstractclassmethod
from collections import namedtuple


class ReplayMemory(ABC):

    @abstractclassmethod
    def insert(self, transitions):
        raise NotImplementedError

    @abstractclassmethod
    def sample(self, batch_size):
        raise NotImplementedError

    @abstractclassmethod
    def __len__(self):
        raise NotImplementedError


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
# Collect sequences of length 80 timesteos, where adjacent overlap by 40 time-steps.
# The sequences never cross boundaries.
# Additionally, we store in the replay the value of the beta_i (intrinsic_factor) used by actor
# as well as the initial recurrent state.
Sequence = namedtuple('Sequence', ('trainsitions', 'initial_recurrent_state', 'intrinsic_factor'))

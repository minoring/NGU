"""Define namedtuples and classes that are act as 'type'"""
from collections import namedtuple

TransitionNamedtuple = namedtuple('Transition', [
    'state', 'prev_action', 'action', 'reward_intrinsic', 'reward_extrinsic', 'reward_augmented',
    'next_state', 'done', 'nstep_reward'
])


class Transition(TransitionNamedtuple):
    __slots__ = ()

    def __new__(cls,
                state,
                prev_action,
                action,
                reward_intrinsic,
                reward_extrinsic,
                reward_augmented,
                next_state,
                done,
                nstep_reward=0.):
        return super(Transition, cls).__new__(cls, state, prev_action, action, reward_intrinsic,
                                              reward_extrinsic, reward_augmented, next_state, done,
                                              nstep_reward)


Sequence = namedtuple('Sequence',
                      ['init_recurr_state', 'transitions', 'intrinsic_factor', 'discount_factor'])

Hiddenstate = namedtuple('Hiddenstate', ['hx', 'cx'])

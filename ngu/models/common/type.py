"""Define namedtuples and classes that are act as 'type'"""


class Transition:

    def __init__(self,
                 state,
                 prev_action,
                 action,
                 reward_intrinsic,
                 reward_extrinsic,
                 reward,
                 next_state,
                 done,
                 nstep_reward=0):
        self.state = state
        self.prev_action = prev_action
        self.action = action
        self.reward_intrinsic = reward_intrinsic
        self.reward_extrinsic = reward_extrinsic
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.nstep_reward = nstep_reward


# Collect sequences of length 80 timesteps, where adjacent overlap by 40 time-steps.
# The sequences never cross boundaries.
# Additionally, we store in the replay the value of the beta_i (intrinsic_factor) used by actor
# as well as the initial recurrent state.
class Sequence:

    def __init__(self, init_recurr_state, transitions, intrinsic_factor):
        self.init_recurr_state = init_recurr_state
        self.transitions = transitions
        self.intrinsic_factor = intrinsic_factor

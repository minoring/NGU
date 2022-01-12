import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ngu.models.r2d2.dueling_lstm import DuelingLSTM
import ngu.utils.pytorch_util as ptu


class R2D2Actor:
    """R2D2 Actor. Actors feed experience into the replay buffer."""

    def __init__(self, envs, num_actors, act_dim, obs_dim, replay_memory, model_hypr):
        """
        Args:
            envs: Vectorized parallel environment.
            num_actors: The number actors collecting experience.
            act_dim: Action dimension.
            obs_dim: Observation dimension (3 channel image shape expected.).
            replay_memory: Replay memory that actors will feed experience.
            model_hypr: Hyperparameters (batch_size, learning_rate, etc.).
        """
        self.envs = envs
        self.num_actors = num_actors
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.replay_memory = replay_memory
        self.model_hypr = model_hypr
        self.N = self.model_hypr['num_mixtures']
        self.eps = self._compute_exploration_factor()
        # Initialize Policy Neural Net
        self.policy = DuelingLSTM(act_dim, obs_dim, model_hypr)
        self.target = DuelingLSTM(act_dim, obs_dim, model_hypr)
        self.target.load_state_dict(self.policy.state_dict())
        self.policy.to(ptu.device)
        self.target.to(ptu.device)

    def collect_step(self):
        """Proceed one step to collect experience."""
        self.envs.reset()
        num_total_step = self.model_hypr['trace_length'] + self.model_hypr['replay_period']

        for t in range(num_total_step):
            pass

    def compute_priorities(self):
        pass

    def _compute_exploration_factor(self):
        """Compute exploration factor (beta_i). The formulation came from the NGU paper Appendix A."""
        beta = self.model_hypr['intrinsic_reward_scale']
        eps = np.zeros((self.num_actors, ))
        for j in range(self.num_actors - 1):
            i = j % self.N - 1
            if i == 0:
                eps[i] = 0
            elif i == self.N - 1:
                eps[i] = beta
            else:
                eps[j] = beta * _sigmoid(10 * (2 * i - (self.N - 2)) / (self.N - 2))


def _sigmoid(self, x):
    return (1 / (1 + math.exp(-x)))

import math

import torch
import numpy as np

import ngu.utils.pytorch_util as ptu
from ngu.models.r2d2.dueling_lstm import DuelingLSTM
from ngu.utils.random_util import choose_among_with_prob


class R2D2Actor:
    """R2D2 Actor. Actors feed experience into the replay buffer."""

    def __init__(self, envs, n_actors, n_act, obs_shape, model_hypr):
        """
        Args:
            envs: Vectorized parallel environment.
            n_actors: The number actors collecting experience.
            n_act: Action dimension.
            obs_shape: Observation dimension (3 channel image shape expected.).
            model_hypr: Hyperparameters (batch_size, learning_rate, etc.).
        """
        self.envs = envs
        self.n_actors = n_actors
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr
        self.N = self.model_hypr['num_mixtures']
        # Initialize exploration factor betas.
        self.explr_beta, self.explr_beta_onehot = self._compute_exploration_factor()
        self.discounts = self._compute_discount_factor()
        # Initialize Policy Neural Net
        self.policy = DuelingLSTM(self.n_actors, n_act, obs_shape, model_hypr)
        self.target = DuelingLSTM(self.n_actors, n_act, obs_shape, model_hypr)
        self.target.load_state_dict(self.policy.state_dict())

        for param in list(self.policy.parameters()) + list(self.target.parameters()):
            param.requires_grad = False

    def reset_hiddenstate_if_done(self, done):
        """If episode done, reset hidden state."""
        self.policy.hx[done.squeeze(-1), :] = torch.zeros(self.policy.hidden_units).to(ptu.device)
        self.policy.cx[done.squeeze(-1), :] = torch.zeros(self.policy.hidden_units).to(ptu.device)

    @torch.no_grad()
    def get_eps_greedy_action(self, obs, prev_act, prev_int_rew, prev_ext_rew):
        obs, prev_act, prev_int_rew, prev_ext_rew, beta_onehot = ptu.to_device(
            (obs, prev_act, prev_int_rew, prev_ext_rew, self.explr_beta_onehot), ptu.device)
        greedy_action = self.policy(obs, prev_act, prev_int_rew, prev_ext_rew,
                                    beta_onehot).argmax(1).cpu()
        random_action = torch.randint(1, self.n_act, (self.n_actors, ))
        # Each actor selects action with their own epsilon.
        action = choose_among_with_prob(greedy_action, random_action, self.explr_beta)
        # Since the returned action is list of tensor, convert it into tensor.
        # And unsqueeze -1 dim to match the interface of environment's step.
        action = torch.tensor(action, dtype=torch.int64)
        action.unsqueeze_(-1)
        return action

    def compute_nstep_reward(self, nstep_buffer, discount):
        """Compute sum of n-step rewards in nstep_buffer.

        Args:
            nstep_buffer: N_STEP x n_actors transitions.
            discount: discount to compute n-step.
        Returns:
            Sum of discounted n-step rewards.
        """
        done_mask = nstep_buffer[0].done  # Mask for if it is done in the middle of n-step.
        r_nstep = torch.zeros((self.n_actors, 1))
        for i in range(self.model_hypr['n_step']):
            done_mask = torch.logical_or(done_mask, nstep_buffer[i].done)
            r_nstep += (discount**i) * (1.0 - done_mask.float()) * nstep_buffer[i].reward_augmented
        return r_nstep

    def _compute_exploration_factor(self):
        """Compute exploration factor (beta_i). The formulation came from the NGU paper Appendix A."""

        def _sigmoid(x):
            return (1 / (1 + math.exp(-x)))

        beta = self.model_hypr['intrinsic_reward_scale']
        explr_beta = torch.zeros((self.n_actors, 1))
        explr_beta_onehot = torch.zeros((self.n_actors, self.N))
        for j in range(self.n_actors):
            i = j % self.N
            if i == 0:
                explr_beta[j] = 0
            elif i == self.N - 1:
                explr_beta[j] = beta
            else:
                explr_beta[j] = beta * _sigmoid(10 * (2 * i - (self.N - 2)) / (self.N - 2))
            explr_beta_onehot[j] = ptu.make_one_hot(explr_beta[j], i, self.N)
        return explr_beta, explr_beta_onehot

    def _compute_discount_factor(self):
        """Compute discount factor for extrinsic reward. Check Appendix A, Equation (4)."""
        discounts = torch.zeros((self.n_actors, 1))
        gamma_max = self.model_hypr['max_discount_extrinsic_reward']
        gamma_min = self.model_hypr['min_discount_extrinsic_reward']
        for j in range(self.n_actors):
            i = j % self.N
            if i == 0:
                discounts[j] = gamma_max
            elif i == self.N - 1:
                discounts[j] = gamma_min
            else:
                discounts[j] = 1 - np.exp(
                    ((self.N - 1 - i) * np.log(1 - gamma_max) + i * np.log(1 - gamma_min)) /
                    (self.N - 1))
        return discounts

    def to(self, device):
        self.policy.to(device)
        self.target.to(device)

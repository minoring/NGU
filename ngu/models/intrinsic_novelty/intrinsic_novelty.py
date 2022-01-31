import torch

from ngu.utils.mpi_util import RunningMeanStd
from ngu.models.intrinsic_novelty.lifelong_novelty import LifelongNovelty
from ngu.models.intrinsic_novelty.episodic_novelty import EpisodicNovelty
import ngu.utils.pytorch_util as ptu


class IntrinsicNovelty:
    """Intrinsic novelty (or exploration bonus) for better exploration.
    This intrinsic novelty is composed of episodic and life-long novelty."""
    def __init__(self, n_actors, n_act, obs_shape, model_hypr, logger):
        self.logger = logger
        self.ll_novel = LifelongNovelty(obs_shape, model_hypr, logger)
        self.epi_novel = EpisodicNovelty(n_actors, n_act, obs_shape, model_hypr, logger)
        self.max_rew = 5.  # NGU paper Equ 1. It was 'L' in the paper.
        self.intrinsic_novel_rms = RunningMeanStd()
        self.update_count = 0

    def compute_intrinsic_novelty(self, obs):
        batch_size = len(obs)
        r_ll = self.ll_novel.compute_lifelong_curiosity(obs)  # alpha_t in the paper.
        r_epi = self.epi_novel.compute_episodic_novelty(obs)
        # Equ. 1.
        # intrinsic reward = episodic reward * min(max(alpha_t, 1), maximum_reward_scaling)).
        less_than_one = r_ll < torch.ones((batch_size, ))
        r_ll[less_than_one] = torch.ones((1, ))
        greater_than_max = r_ll > torch.full((batch_size, ), self.max_rew)
        r_ll[greater_than_max] = torch.full((1, ), self.max_rew)

        intrinsic_novelty = (r_epi * r_ll).unsqueeze(-1)
        self.intrinsic_novel_rms.update(intrinsic_novelty.numpy())

        return intrinsic_novelty

    def reset_memory_if_done(self, done):
        """Reset the agent's memory if it is done."""
        done = done.to(ptu.device)
        self.epi_novel.episodic_memory[:, done.squeeze(-1), :] = torch.zeros(
            (self.epi_novel.capacity, 1, self.epi_novel.controllable_state_dim), device=ptu.device)

    def to(self, device):
        self.ll_novel.to(device)
        self.epi_novel.to(device)

    def step(self, timestep_seq):
        """Update parameters of intrinsic novelty.

        Args:
            timestep_seq: Last N frames of the sampled sequences to train the action prediction network and RND.
        """
        self.ll_novel.step(timestep_seq)
        self.epi_novel.step(timestep_seq)
        self.update_count += 1

        self.logger.log_scalar('IntrinsicNoveltyMean', self.intrinsic_novel_rms.mean,
                               self.update_count)
        self.logger.log_scalar('IntrinsicNoveltyVar', self.intrinsic_novel_rms.var,
                               self.update_count)

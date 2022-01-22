import torch

from ngu.models.intrinsic_novelty.lifelong_novelty import LifelongNovelty
from ngu.models.intrinsic_novelty.episodic_novelty import EpisodicNovelty
from ngu.models import model_hypr


class IntrinsicNovelty:
    """Intrinsic novelty (or exploration bonus) for better exploration.
    This intrinsic novelty is composed of episodic and life-long novelty."""

    def __init__(self, n_act, obs_shape=(1, 84, 84), model_hypr=model_hypr):
        self.ll_novel = LifelongNovelty(obs_shape, model_hypr)
        self.epi_novel = EpisodicNovelty(n_act, obs_shape, model_hypr)
        self.max_rew = 5.

    def compute_intrinsic_novelty(self, obs, obs_next):
        batch_size = len(obs)

        r_ll = self.ll_novel.compute_lifelong_curiosity(obs)  # alpha_t in the paper.
        r_epi = self.epi_novel.compute_episodic_novelty(obs, obs_next)
        # Equ. 1.
        # intrinsic reward = episodic reward * min(max(alpha_t, 1), maximum_reward_scaling)).
        less_than_one = r_ll < torch.ones((batch_size, ))
        r_ll[less_than_one] = torch.ones((1, ))
        greater_than_max = r_ll > torch.full((batch_size, ), self.max_rew)
        r_ll[greater_than_max] = torch.full((1, ), self.max_rew)

        return (r_epi * r_ll).unsqueeze(-1)

    def to(self, device):
        self.ll_novel.to(device)
        self.epi_novel.to(device)

    def step(self, batch_sequences):
        """Update parameters of intrinsic novelty."""
        self.ll_nove.step()
        self.epi_novel.step()

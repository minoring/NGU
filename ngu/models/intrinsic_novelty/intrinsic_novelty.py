from ngu.models.intrinsic_novelty.lifelong_novelty import LifelongNovelty
from ngu.models.intrinsic_novelty.episodic_novelty import EpisodicNovelty
from ngu.models import model_hypr


class IntrinsicNovelty:
    """Intrinsic novelty (or exploration bonus) for better exploration.
    This intrinsic novelty is composed of episodic and life-long novelty."""

    def __init__(self, act_dim, obs_dim=(1, 84, 84), model_hypr=model_hypr):
        self.ll_novel = LifelongNovelty(model_hypr)
        self.epi_novel = EpisodicNovelty(act_dim, obs_dim, model_hypr)
        self.max_rew = 5

    def compute_intrinsic_novelty(self, obs):
        # Equ 1.
        r_epi = self.epi_novel.compute_episodic_novelty(obs)
        r_ll = self.self.ll_novel.compute_lifelong_curiosity(obs)  # alpha_t in the paper.
        return r_epi * min(max(r_ll, 1), self.max_rew)

    def to(self, device):
        self.ll_novel.to(device)
        self.epi_novel.to(device)

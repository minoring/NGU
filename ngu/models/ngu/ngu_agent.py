from ngu.models.r2d2 import R2D2Learner, R2D2Actor
from ngu.models.r2d2.replay_memory import PrioritizedReplayMemory
from ngu.models.intrinsic_novelty import IntrinsicNovelty


class NGUAgent:
    """"""

    def __init__(self, envs, num_actors, act_dim, obs_dim, model_hypr):
        self.envs = envs
        self.num_actors = num_actors
        self.model_hypr = model_hypr

        # Initialize Policy (R2D2).
        self.r2d2_memory = PrioritizedReplayMemory(self.model_hypr['replay_capacity'])
        self.r2d2_learner = R2D2Learner(act_dim, obs_dim, self.r2d2_memory, model_hypr)
        self.r2d2_actor = R2D2Actor(envs, num_actors, act_dim, obs_dim, self.r2d2_memory,
                                    model_hypr)
        # Initialize Intrinsic Novelty (episodic, life-long module).
        self.intrinsic_novelty = IntrinsicNovelty(act_dim, obs_dim, model_hypr)

    def to(self, device):
        self.r2d2_learner.to(device)
        self.r2d2_actor.to(device)
        self.intrinsic_novelty.to(device)

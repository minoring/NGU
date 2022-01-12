"""Hyperparameters for NGU, RND model.

Take a look at
- Appendix F (Badia and Sprechmann et al., Never Give Up: Learning Directed Exploration Strategies, ICLR, 2020.)
- Appendix A.3 (Burda and Edwards et al., Exploration by Random Network Distillation, ICLR, 2019.
"""
import numpy as np
import torch

from ngu.models.utils import reward_transform, reward_transform_inverted

model_hypr = dict(
    # NGU
    num_mixtures=30,  # Number of mixtures N.
    # Optimizer.
    learning_rate_r2d2=0.0001,
    learning_rate_rnd=0.0005,
    adam_epsilon=0.0001,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_clip_norm=40,
    batch_size=64,
    # R2D2.
    trace_length=80,
    replay_period=40,  # TODO(minho): Is this burn-in???
    r2d2_reward_transform=reward_transform,
    r2d2_reward_transformation_inverted=reward_transform_inverted,
    actor_update_period=100,  # # Actor parameter update interval in terms of environment step.
    priority_exponent=1,
    intrinsic_reward_scale=0.3,  # Beta
    # Episodic memory.
    episodic_memory_capacity=30000,
    kernel_epsilon=0.0001,
    num_neighbors=10,  # How many neighbors are used, Ker num in the paper.
    cluster_distance=0.008,
    kernel_pseudo_counts_constant=0.001,
    kernel_maximum_similarity=8,
    # Replay Memory.
    replay_priority_exponent=0.9,
    replay_capacity=5000000,  # 5M
    minimum_sequences_to_start_replay=6250,
)

"""Hyperparameters for NGU, R2D2, RND model.

Take a look at
- Appendix F (Badia and Sprechmann et al., Never Give Up: Learning Directed Exploration Strategies, ICLR, 2020.)
- Appendix A.3 (Burda and Edwards et al., Exploration by Random Network Distillation, ICLR, 2019.
"""
from ngu.models.utils import reward_transform, reward_transform_inverted

model_hypr = dict(
    # NGU
    num_mixtures=32,  # Number of mixtures N.
    max_discount_intrinsic_reward=0.99,
    max_discount_extrinsic_reward=0.997,
    min_discount_extrinsic_reward=0.99,
    init_obs_step=128, # Number of steps to initialize observation normalization.
    # Optimizer.
    learning_rate_r2d2=0.0001,
    learning_rate_rnd=0.0005,
    learning_rate_action_prediction=0.0005,
    adam_epsilon=0.0001,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_clip_norm=40,
    batch_size=64,
    action_prediction_l2_weight=0.00001,
    # R2D2.
    trace_length=80,
    replay_period=40,
    r2d2_reward_transform=reward_transform,
    r2d2_reward_transformation_inverted=reward_transform_inverted,
    actor_update_period=
    100,  # Actor parameter update interval in terms of the number of parameter updates.
    priority_exponent=0.9,  # eta in R2D2.
    importance_sampling_exponent=0.6,
    intrinsic_reward_scale=0.3,  # Beta
    target_q_update_period=1500,
    n_step=5,  # n-step TD error.
    beta_decay=1000000, # 1M
    beta0=0.4,  # Beta0 of prioritized replay memory.
    # The number of steps per sequence collect.
    # This number is roughly "time to collect sequence" / "time to update parameters".
    step_per_collect=5,
    # Episodic memory.
    # Rough calculation of required memory.
    # **********
    # memory_capacity x num_actors x controllable_state x bits / byte / Giga
    # 30000 x 64 x 32 x 32 / 8 / 10^9 = 0.245 GB
    # **********
    episodic_memory_capacity=30000,
    kernel_epsilon=0.0001,
    num_neighbors=10,  # How many neighbors are used, Ker num in the paper.
    cluster_distance=0.008,
    kernel_pseudo_counts_constant=0.001,
    kernel_maximum_similarity=8,
    # Replay Memory.
    replay_priority_exponent=0.9,
    remove_to_fit_interval=
    100,  # The number of learning step before removing sequences that exceed memory capacity.
    # Rough calculation for required memory size.
    # **********
    # memory_capacity x sequence_length x transition_size x bits / byte / Giga
    # 5M x 120 x (1 x 84 x 84 x 2) x 32 / 8 / 10^9 = 16934 GB.
    # **********
    # replay_capacity=5000000 // 120,  # 5M / SEQUENCE_LENGTH
    replay_capacity=5000000 // 120,
    minimum_sequences_to_start_replay=6250,
    # Last N frames of the sampled sequences to trian the action prediction networkand RND.trinsic novelty.
    num_frame_intrinsic_train=5,
)

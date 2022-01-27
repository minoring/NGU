"""Appendix F.4 Table 8 (Badia and Sprechmann et al., 2020)"""
atari_env_hypr = dict(
    max_episode_steps=18000,  # NGU uses 30 minutes, here we use 18K frames to approximate this.
    num_action_repeats=4,
    num_stacked_frame=1,  # No frame stack.
    zero_discount_on_life_loss=False,
    random_noops_range=30,
    sticky_actions=False,
    frames_max_pooled=3,  # or 4
    grayscaled=True,  # False for rgb
    full_action_set=True,
    # RND
    rnd_obs_clipping_factor=5,
)

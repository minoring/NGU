"""NGU paper appendix Table 8: Atari pre-processing hyperparameters."""
preproc_conf = {
    'max_episode_steps': 4500, # Paper says 30 minutes, use 4500 frames to approximate.
    'num_action_repeats': 4,
    'num_stacked_frame': 1, # No frame stack.
    'zero_discount_on_life_loss': False,
    'random_noops_range': 30,
    'sticky_actions': False,
    'frames_max_pooled': 3, # or 4
    'grayscaled': True, # False for rgb
    'full_action_set': True
}

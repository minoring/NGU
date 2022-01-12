"""Hyperparameters related to environments.s"""
# Preprocessing hyperparameters for OpenAI Gym Classic Control
classic_control_env_hypr = dict(
    num_stacked_frame = 1, # No frame stack.
    frames_max_pooled = 3, # or 4
    grayscaled = True, # False for rgb
    full_action_set = True
)

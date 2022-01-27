"""Utility functions for environment."""
import os
import time

import gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ngu.envs.wrapper import VecNormalize, VecPyTorch, VecPyTorchFrameStack
from ngu.envs.atari.atari_utils import make_atari_env
from ngu.envs.classic_control.classic_control_utils import make_classic_control_env
from ngu.envs.atari import atari_env_hypr
from ngu.envs.classic_control import classic_control_env_hypr


def make_vec_envs(env_id, num_env, seed, device, monitor_root):
    """Create a wrapped, preprocessed, vectorized parallel environments of Atari.

    Args:
        env_id: OpenAI Gym environment ID.
        num_env: The number of parallel environments.
        seed: Random seed for the environments.
        device: PyTorch tensor device.
        monitor_root: Directory to save monitor files.
    Returns:
        Vectorized parallel environment.
    """
    env_creator = get_env_maker(env_id)
    env_hypr = get_env_hypr(env_id)

    # Make directory to save monitor csv files.
    monitor_dir = os.path.join(monitor_root, env_id + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(monitor_dir)
    envs = [env_creator(env_id, seed, i, env_hypr, monitor_dir) for i in range(num_env)]
    envs = SubprocVecEnv(envs) if len(envs) > 1 else DummyVecEnv(envs)
    envs = VecNormalize(envs, norm_reward=False, clip_obs=env_hypr['rnd_obs_clipping_factor'])
    envs = VecPyTorch(envs, device)

    if env_hypr['num_stacked_frame'] > 1:
        envs = VecPyTorchFrameStack(envs, env_hypr['num_stacked_frame'], device)

    assert isinstance(envs.action_space,
                      gym.spaces.Discrete), "This project only support discrete action space."

    return envs


def get_env_maker(env_id):
    if is_atari(env_id):
        return make_atari_env
    elif is_classic_control(env_id):
        return make_classic_control_env
    else:
        raise ValueError(f"{env_id} is not supported in this project.")


def get_env_hypr(env_id):
    if is_atari(env_id):
        return atari_env_hypr
    elif is_classic_control(env_id):
        return classic_control_env_hypr
    else:
        raise ValueError(f"{env_id} is not supported in this project.")


def is_atari(env_id):
    env_list = ['MontezumaRevenge']
    return list_item_in_s(env_list, env_id)


def is_classic_control(env_id):
    env_list = ['Acrobot', 'CartPole', 'MountainCar', 'Pendulum']
    return list_item_in_s(env_list, env_id)


def list_item_in_s(l, s):
    """Check any of item in the list l is 'in' string s"""
    for item in l:
        if item in s:
            return True
    return False

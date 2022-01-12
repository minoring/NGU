"""Utility functions for environment."""
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ngu.envs.wrapper import VecNormalize, VecPyTorch, VecPyTorchFrameStack
from ngu.envs.atari.atari_utils import make_atari_env
from ngu.envs.classic_control.classic_control_utils import make_classic_control_env


def make_vec_envs(env_id, num_env, seed, device, env_hypr, logger=None):
    """Create a wrapped, preprocessed, vectorized parallel environments of Atari.

    Args:
        env_id: OpenAI Gym environment ID.
        num_env: The number of parallel environments.
        seed: Random seed for the environments.
        device: PyTorch tensor device.
        env_hypr: Hyperparams for preprocessing.
    Returns:
        Vectorized parallel environment.
    """
    env_creator = get_env_maker(env_id)
    envs = [env_creator(env_id, seed, i, env_hypr) for i in range(num_env)]
    envs = SubprocVecEnv(envs) if len(envs) > 1 else DummyVecEnv(envs)
    envs = VecNormalize(envs, norm_reward=False, clip_obs=env_hypr['rnd_obs_clipping_factor'])
    envs = VecPyTorch(envs, device)

    if env_hypr['num_stacked_frame'] > 1:
        envs = VecPyTorchFrameStack(envs, env_hypr['num_stacked_frame'], device)

    return envs


def get_env_maker(env_id):
    if is_atari(env_id):
        return make_atari_env
    elif is_classic_control(env_id):
        return make_classic_control_env
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

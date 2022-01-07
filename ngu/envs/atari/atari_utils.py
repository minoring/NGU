import gym

from stable_baselines3.common.atari_wrappers import (ClipRewardEnv, FireResetEnv, MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ngu.envs.atari.atari_wrapper import (StickyActionEnv, TransposeImage, VecNormalize, VecPyTorch,
                                          VecPyTorchFrameStack)


def make_atari_env(env_id, seed, rank, hypr):

    def _thunk():
        env = gym.make(env_id)
        assert 'NoFrameskip' in env.spec.id
        env.seed(seed + rank)  # Each parellel environment will have different seed.

        env._max_episode_steps = hypr['max_episode_steps'] * hypr['num_action_repeats']
        env = NoopResetEnv(env, noop_max=hypr['random_noops_range'])  # For random initial state.
        env = MaxAndSkipEnv(env, skip=hypr['num_action_repeats'])  # Frame skip.

        if hypr['sticky_actions']:
            env = StickyActionEnv(env)  # Add some randomness by sticky action.

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)  # Take action 'FIRE' if needed in initial state.

        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 3  # 3 channel image expected.
        env = WarpFrame(env, width=84, height=84)
        env = ClipRewardEnv(env)  # Reward clipping to have same hyperparameter across many games.
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_atari_vec_envs(env_id, num_env, seed, device, hypr):
    """Create a wrapped, preprocessed, vectorized parallel environments of Atari.

    Args:
        env_id: OpenAI Gym environment ID.
        num_env: The number of parallel environments.
        seed: Random seed for the environments.
        device: PyTorch tensor device.
        hypr: Hyperparams for preprocessing.
    Returns:
        Vectorized parallel environment.
    """
    envs = [make_atari_env(env_id, seed, i, hypr) for i in range(num_env)]
    envs = SubprocVecEnv(envs) if len(envs) > 1 else DummyVecEnv(envs)
    envs = VecNormalize(envs, norm_reward=False)
    envs = VecPyTorch(envs, device)

    if hypr['num_stacked_frame'] > 1:
        envs = VecPyTorchFrameStack(envs, hypr['num_stacked_frame'], device)

    return envs

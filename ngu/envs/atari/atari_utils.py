import os

import gym
from stable_baselines3.common.atari_wrappers import (FireResetEnv, MaxAndSkipEnv, NoopResetEnv,
                                                     WarpFrame)

from ngu.envs.wrapper import (DummyMontezumaInfoWrapper, MontezumaInfoWrapper, StickyActionEnv,
                              TransposeImage, Monitor)


def make_atari_env(env_id, seed, rank, env_hypr, monitor_dir, video_dir):

    def _thunk():
        env = gym.make(env_id)
        assert 'NoFrameskip' in env.spec.id
        env.seed(seed + rank)  # Each parellel environment will have different seed.

        env._max_episode_steps = env_hypr['max_episode_steps'] * env_hypr['num_action_repeats']
        if env_hypr['random_noops_range'] > 0:
            env = NoopResetEnv(env,
                               noop_max=env_hypr['random_noops_range'])  # For random initial state.
        env = MaxAndSkipEnv(env, skip=env_hypr['frames_max_pooled'])  # Frame skip.

        if "Montezuma" in env_id or "Pitfall" in env_id:
            env = MontezumaInfoWrapper(env, room_address=3 if "Montezuma" in env_id else 1)
        else:
            env = DummyMontezumaInfoWrapper(env)

        if env_hypr['sticky_actions']:
            env = StickyActionEnv(env)  # Add some randomness by sticky action.

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)  # Take action 'FIRE' if needed in initial state.

        monitor_path = os.path.join(monitor_dir, f'{rank}')
        # Log episodic reward, episode length and misc information.
        env = Monitor(env, monitor_path, allow_early_resets=True)

        if video_dir is not None:
            env = gym.wrappers.Monitor(env, video_dir)

        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 3  # 3 channel image expected.
        env = WarpFrame(env, width=84, height=84)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk

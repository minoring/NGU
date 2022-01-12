import gym

from stable_baselines3.common.atari_wrappers import WarpFrame

from ngu.envs.wrapper import TransposeImage


def make_classic_control_env(env_id, seed, rank, env_hypr):

    def _thunk():
        env = gym.make(env_id)
        assert 'NoFrameskip' in env.spec.id
        env.seed(seed + rank)  # Each parellel environment will have different seed.

        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 3  # 3 channel image expected.
        env = WarpFrame(env, width=84, height=84)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk

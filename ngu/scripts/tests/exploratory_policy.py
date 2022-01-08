"""For sanity check of exploratory policy, analyze single exploratory policy in montezuma revenge environment.
This experiment is purely exploratory, with no external reward.
Performance is measured by the number of room visited by the agent.
"""
import torch

from ngu.utils.args import get_args
from ngu.envs.atari.atari_utils import make_atari_vec_envs
from ngu.envs.atari.config import preproc_conf
from ngu.utils.random_util import set_global_seed


def main():
    args = get_args()
    set_global_seed(args.seed)

    env = make_atari_vec_envs(env_id='MontezumaRevengeNoFrameskip-v4',
                              num_env=args.num_env,
                              seed=args.seed,
                              device='cpu',
                              hypr=preproc_conf)
    env.reset()
    while True:
        obs, rew, done, info = env.step(torch.zeros((args.num_env, 1),
                                                    dtype=torch.uint8))  # Dummmy action.


if __name__ == '__main__':
    main()

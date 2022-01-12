"""For sanity check of exploratory policy, analyze single exploratory policy in montezuma revenge environment.
This experiment is purely exploratory, with no external reward.
The intrinsic novelty is composed by episodic novelty and life-long novelty.
Performance is measured by mean episodic return, and the number of rooms the agent finds over the training run.
"""
import torch

from ngu.utils.args import get_args
from ngu.envs.utils import make_vec_envs
from ngu.envs.atari.atari_env_hypr import atari_env_hypr
from ngu.utils.random_util import set_global_seed


def main():
    args = get_args()
    set_global_seed(args.seed)

    env = make_vec_envs(env_id='MontezumaRevengeNoFrameskip-v4',
                        num_env=args.num_actors,
                        seed=args.seed,
                        device='cpu',
                        hypr=atari_env_hypr)
    env.reset()
    while True:
        obs, rew, done, info = env.step(torch.zeros((args.num_actors, 1),
                                                    dtype=torch.uint8))  # Dummmy action.


if __name__ == '__main__':
    main()

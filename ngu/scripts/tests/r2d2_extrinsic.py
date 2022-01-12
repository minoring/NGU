"""Train R2D2 agent, only driven by extrinsic reward for sanity check."""
import torch
from ngu.models.r2d2.r2d2_actor import R2D2Actor
from ngu.models.r2d2.r2d2_learner import R2D2Learner

import ngu.utils.pytorch_util as ptu
from ngu.utils.args import get_args
from ngu.envs.utils import make_vec_envs
from ngu.envs.atari import atari_env_hypr
from ngu.utils.random_util import set_global_seed
from ngu.models import model_hypr
from ngu.models.r2d2.replay_memory import UniformReplayMemory


def main():
    args = get_args()
    set_global_seed(args.seed)
    ptu.init_device()

    # Initialize the environment.
    env = make_vec_envs(env_id='ALE/Breakout-v5',
                        num_env=args.num_env,
                        seed=args.seed,
                        device=ptu.device,
                        hypr=atari_env_hypr)
    env.reset()
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape

    # Initialize replay memory that actors and a learner will share.
    # Convert it into PrioritizedReplayMemory.
    replay_memory = UniformReplayMemory(model_hypr['replay_capacity'])
    # Initialize actor and learner.
    actor = R2D2Actor(act_dim, obs_dim, replay_memory, model_hypr, env, args.num_env)
    learner = R2D2Learner(act_dim, obs_dim, replay_memory, model_hypr)

    while True:
        actor.collect_step()


if __name__ == '__main__':
    main()

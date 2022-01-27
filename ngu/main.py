"""Train NGU model."""
import os
import time
from itertools import count

import torch

import ngu.utils.pytorch_util as ptu
from ngu.models.ngu import NGUAgent
from ngu.utils.args import get_args
from ngu.envs.utils import make_vec_envs
from ngu.utils.random_util import set_global_seed
from ngu.models import model_hypr
from ngu.utils.logger import Logger


def main():
    args = get_args()
    set_global_seed(args.seed)
    ptu.init_device()

    current_time = time.strftime("%d-%m-%Y_%H-%M-%S")
    log_root = os.path.join('log', f'{args.env_id}_{current_time}')
    # Create directory to save trained model.
    trained_model_dir = os.path.join(log_root, 'trained_model')
    os.makedirs(trained_model_dir)

    # Create vectorized environment.
    envs = make_vec_envs(env_id=args.env_id,
                         num_env=args.n_actors,
                         seed=args.seed,
                         device='cpu',
                         log_root=log_root)
    n_act = envs.action_space.n
    obs_shape = envs.observation_space.shape

    logger = Logger(args.env_id, log_root)
    ngu_agent = NGUAgent(envs, args.n_actors, n_act, obs_shape, model_hypr, logger)
    ngu_agent.to(ptu.device)

    ngu_agent.collect_minimum_sequences()  # Collect minimum experience to run replay.
    for param_update_count in count(1):
        ngu_agent.step()  # Update parameters single step.
        ngu_agent.collect_sequence()  # Each parallel actors collect a sequence.

        if param_update_count % args.model_save_interval == 0:
            print(f"Saving trained model. [learning step: {param_update_count}]")
            model_name = f"{args.env_id}_{param_update_count}_model.pt"
            save_path = os.path.join(trained_model_dir, model_name)
            torch.save(ngu_agent.r2d2_learner.policy.state_dict(), save_path)


if __name__ == '__main__':
    main()

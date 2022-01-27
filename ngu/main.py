"""Train NGU model."""
from itertools import count

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
    # Create vectorized environment.
    envs = make_vec_envs(env_id=args.env_id,
                         num_env=args.n_actors,
                         seed=args.seed,
                         device='cpu',
                         monitor_root=args.monitor_root)
    n_act = envs.action_space.n
    obs_shape = envs.observation_space.shape

    logger = Logger(args.env_id, args.log_csv_path, args.log_dir)
    ngu_agent = NGUAgent(envs, args.n_actors, n_act, obs_shape, model_hypr, logger)
    ngu_agent.to(ptu.device)

    ngu_agent.collect_minimum_sequences()  # Collect minimum experience to run replay.
    for param_update_count in count():
        ngu_agent.collect_sequence()  # Each parallel actors collect a sequence.
        ngu_agent.step(param_update_count)  # Update parameters single step.


if __name__ == '__main__':
    main()

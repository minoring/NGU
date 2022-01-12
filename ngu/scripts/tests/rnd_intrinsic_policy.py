"""To test RND intrinsic bonus, analyze to performance on Montezuma Revenge environment.
The policy is driven only by intrinsic bonus, in the absence of any extrinsic reward.
Performance is measured by mean episodic return, and the number of rooms the agent finds over the training run.
"""
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim

import ngu.utils.pytorch_util as ptu
from ngu.utils.args import get_args
from ngu.envs.utils import make_vec_envs
from ngu.envs.atari import atari_env_hypr
from ngu.utils.random_util import set_global_seed
from ngu.models.lifelong_novelty import RNDAgent
from ngu.models import model_hypr


def main():
    args = get_args()
    set_global_seed(args.seed)
    ptu.init_device()

    # Create environment. Tale a look at RND Appendix A.3 for the environment hyperparameters.
    env_hypr = atari_env_hypr.copy()
    env_hypr['sticky_actions'] = True
    env_hypr['random_noops_range'] = 0
    env = make_vec_envs(
        env_id='MontezumaRevengeNoFrameskip-v4',
        num_env=32,  # RND Appendix A.4
        seed=args.seed,
        device=ptu.device,
        env_hypr=env_hypr)

    act_dim = env.action_space.n  # Atari has discrete action space.
    obs_dim = env.observation_space.shape
    rnd_agent = RNDAgent(act_dim, obs_dim)
    rnd_agent.to(ptu.device)

    # Initialize optimizer
    optimizer = optim.Adam(rnd_agent.parameters(),
                           lr=model_hypr['learning_rate_rnd'],
                           betas=(model_hypr['adam_beta1'], model_hypr['adam_beta2']),
                           eps=model_hypr['adam_epsilon'])

    obs = env.reset()
    for num_param_updates in count():
        act = rnd_agent.get_action(obs)  # Intrinsically a- policy.
        obs, rew, done, info = env.step(act)

        # Update single step
        predictor_act, target_act = rnd_agent(obs)
        loss = F.mse_loss(predictor_act, target_act)
        optimizer.zero_grad()
        loss.backward()
        import pdb
        pdb.set_trace()
        optimizer.step()

        # TODO(minho): How to calculate return and visited room for parallel environment.
        print(f"The number of parameter updates: {num_param_updates}")
        print("Loss: {:.4f}".format(loss.item()))
        print("Num visited rooms: {}")


if __name__ == '__main__':
    main()

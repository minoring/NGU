import torch

from ngu.envs.utils import make_vec_envs


class RecordRunner:
    """Evaluative runner that record the video."""
    def __init__(self, env_id, seed, log_root):
        self.env = make_vec_envs(env_id,
                                 num_env=1,
                                 seed=seed,
                                 device='cpu',
                                 log_root=log_root,
                                 record_video=True)

    @torch.no_grad()
    def record(self, agent):
        org_hidden_size = len(agent.r2d2_learner.policy.hx)
        agent.r2d2_learner.policy.reset_hidden_state(hidden_size=1)
        # Dummy previous obs, action, reward.
        obs = self.env.reset()
        action = torch.zeros((1, 1))
        rew = torch.zeros((1, 1))

        done = False
        while not done:
            action = agent.get_greedy_action(obs, action, rew)
            obs, rew, done, info = self.env.step(action)

            if done.item():
                break

        agent.r2d2_learner.policy.reset_hidden_state(hidden_size=org_hidden_size)

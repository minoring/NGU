import torch.nn as nn
import torch.optim as optim

from ngu.models.r2d2.dueling_lstm import DuelingLSTM


class R2D2Learner:
    """R2D2 Learner. Learner is trained consuming experiences collected by multiple actors.
    Modified version of Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., 2019),
    introduced in NGU paper.
    """

    def __init__(self, act_dim, obs_dim, replay_memory, model_hypr):
        """
        Args:
            act_dim: Action dimension
            obs_dim: Observation dimension (3 channel image shape expected.)
            replay_memory: Replay memory containing experiences learner will train.
            model_hypr: Hyperparameters (batch_size, learning_rate, etc.)
        """
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.replay_memory = replay_memory
        self.model_hypr = model_hypr
        # Initialize Policy Neural Net
        self.policy = DuelingLSTM(act_dim, obs_dim, model_hypr)
        self.target = DuelingLSTM(act_dim, obs_dim, model_hypr)
        self.target.load_state_dict(self.policy.state_dict())
        # TODO(minho): Beta decay...?
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=model_hypr['learning_rate_r2d2'],
                                    betas=(model_hypr['adam_beta1'], model_hypr['adam_beta2']),
                                    eps=model_hypr['adam_epsilon'])
        # TODO(minho): Make sure grad clip.
        self.update_count = 0  # Count how many times the policy updated.

    def update(self):
        """Update policy parameters given memory is collected."""
        if len(self.replay_memory) < self.model_hypr['minimum_sequences_to_start_replay']:
            return

        self.update_count += 0

    def to(self, device):
        self.policy.to(device)
        self.target.to(device)
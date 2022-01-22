import torch.nn as nn
import torch.optim as optim

from ngu.models.r2d2.dueling_lstm import DuelingLSTM


class R2D2Learner:
    """R2D2 Learner. Learner is trained consuming experiences collected by multiple actors.
    Modified version of Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., 2019),
    introduced in NGU paper.
    """

    def __init__(self, n_act, obs_shape, replay_memory, model_hypr):
        """
        Args:
            n_act: Action dimension.
            obs_shape: Observation dimension (3 channel image shape expected).
            replay_memory: Prioritized replay memory that actors fill in experience.
            model_hypr: Hyperparameters (batch_size, learning_rate, etc.).
        """
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.replay_memory = replay_memory
        self.model_hypr = model_hypr
        # Initialize Policy Neural Net
        self.policy = DuelingLSTM(n_act, obs_shape, model_hypr)
        self.target = DuelingLSTM(n_act, obs_shape, model_hypr)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=model_hypr['learning_rate_r2d2'],
                                    betas=(model_hypr['adam_beta1'], model_hypr['adam_beta2']),
                                    eps=model_hypr['adam_epsilon'])
        # TODO(minho): Make sure grad clip.
        self.update_count = 0  # Count how many times the policy updated.

    def update(self, batch_sequences):
        """Update policy parameters given memory is collected.

        Args:
            batch_sequences: Batch of sequence to train.
        """
        # TODO(minho): Implement learner.
        loss = self.loss(batch_sequences)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.model_hypr['target_q_update_period']:
            self.target.load_state_dict(self.policy.state_dict())

    def to(self, device):
        self.policy.to(device)
        self.target.to(device)

    def loss(self, samples):
        pass

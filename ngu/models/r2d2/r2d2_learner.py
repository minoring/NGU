import torch.nn as nn
import torch.optim as optim

from ngu.models.r2d2.dueling_lstm import DuelingLSTM
from ngu.utils.mpi_util import RunningMeanStd


class R2D2Learner:
    """R2D2 Learner. Learner is trained consuming experiences collected by multiple actors.
    Modified version of Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., 2019),
    as introduced in the NGU paper.
    """
    def __init__(self, n_act, obs_shape, replay_memory, model_hypr, logger):
        """
        Args:
            n_act: Action dimension.
            obs_shape: Observation dimension (3 channel image shape expected).
            replay_memory: Prioritized replay memory that actors fill in experience.
            model_hypr: Hyperparameters (batch_size, learning_rate, etc.).
            logger: Logger.
        """
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.replay_memory = replay_memory
        self.model_hypr = model_hypr
        self.logger = logger
        # Initialize Policy Neural Net
        self.policy = DuelingLSTM(self.model_hypr['batch_size'], n_act, obs_shape, model_hypr)
        self.target = DuelingLSTM(self.model_hypr['batch_size'], n_act, obs_shape, model_hypr)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=model_hypr['learning_rate_r2d2'],
                                    betas=(model_hypr['adam_beta1'], model_hypr['adam_beta2']),
                                    eps=model_hypr['adam_epsilon'])
        for param in self.target.parameters():
            param.requires_grad = False

        self.update_count = 0
        self.r2d2_loss_rms = RunningMeanStd(use_mpi=False)

    def step(self, td_errors, weights):
        """Update policy parameters given memory is collected.

        Args:
            td_errors: TD-error between the policy and the target.
            weights: Weights of importance sampling.
        """
        loss = (td_errors.pow(2).mean(dim=0) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.model_hypr['adam_clip_norm'])
        self.optimizer.step()
        self.update_count += 1
        self.r2d2_loss_rms.update(loss.item())
        self.logger.log_scalar('R2D2Loss', self.r2d2_loss_rms.mean, self.update_count)
        self.logger.log_scalar('R2D2ISWeightMean', weights.mean().item(), self.update_count)
        self.logger.log_scalar('R2D2ISWeightVar', weights.var().item(), self.update_count)

    def to(self, device):
        self.policy.to(device)
        self.target.to(device)

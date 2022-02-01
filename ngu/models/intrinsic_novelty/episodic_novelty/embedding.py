import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

import ngu.utils.pytorch_util as ptu
from ngu.models.utils import conv2d_out
from ngu.models.common import weight_init


class Embedding(nn.Module):
    """Mapping the current observation to a learned representation that the NGU paper refer to as controllable state.
    This can be thought as Siamese network to predict the action taken by the agent to go from one observation to the next.
    """
    def __init__(self, n_act, obs_shape, ctrl_state_dim, model_hypr, logger):
        super(Embedding, self).__init__()
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.ctrl_state_dim = ctrl_state_dim
        self.model_hypr = model_hypr
        self.logger = logger

        h = conv2d_out(conv2d_out(conv2d_out(self.obs_shape[1], 4, 4), 4, 2), 3, 1)
        w = conv2d_out(conv2d_out(conv2d_out(self.obs_shape[2], 4, 4), 4, 2), 3, 1)
        # Appendix H.1 for more details about the architecture.
        self.siamese = nn.Sequential(
            weight_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)), nn.ReLU(),
            weight_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            weight_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.Flatten(),
            weight_init(nn.Linear(h * w * 64, self.ctrl_state_dim)))
        self.h = nn.Sequential(
            # 32 for current observation, the other 32 for next observation.
            nn.ReLU(),
            weight_init(nn.Linear(self.ctrl_state_dim * 2, 128)),
            nn.ReLU(),
            weight_init(nn.Linear(128, n_act), init.calculate_gain('linear')))

        self.optimizer = optim.Adam(list(self.siamese.parameters()) + list(self.h.parameters()),
                                    lr=model_hypr['learning_rate_action_prediction'],
                                    betas=(model_hypr['adam_beta1'], model_hypr['adam_beta2']),
                                    eps=model_hypr['adam_epsilon'],
                                    weight_decay=self.model_hypr['action_prediction_l2_weight'])
        self.criterion = nn.CrossEntropyLoss()
        self.update_count = 0

    def forward(self, obs_curr):
        return self.siamese(obs_curr)

    def step(self, timestep_seq):
        # Trained via maximum likelihood.
        loss_avg = 0.
        for t in range(len(timestep_seq)):
            f_curr = self(timestep_seq[t].state.to(ptu.device))
            f_next = self(timestep_seq[t].next_state.to(ptu.device))
            action = timestep_seq[t].action.to(ptu.device).squeeze(-1)
            inverse_dynamic = self.h(torch.cat((f_curr, f_next), dim=1))
            self.optimizer.zero_grad()
            loss = self.criterion(inverse_dynamic, action)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.siamese.parameters()) + list(self.h.parameters()),
                self.model_hypr['adam_clip_norm'])
            self.optimizer.step()
            loss_avg = loss_avg + (1 / (t + 1)) * (loss.item() - loss_avg)
        print("Action Prediction Loss: {:.4f}".format(loss_avg))
        self.update_count += 1
        self.logger.log_scalar('ActionpredictionLoss', loss_avg, self.update_count)

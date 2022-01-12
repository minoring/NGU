import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ngu.models import model_hypr
from ngu.models.utils import conv2d_out
from ngu.models.common import weight_init


class DuelingLSTM(nn.Module):
    """Dueling LSTM Model of R2D2"""

    def __init__(self, act_dim, obs_dim=(1, 84, 84), model_hypr=model_hypr):
        super(DuelingLSTM, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.model_hypr = model_hypr

        # Hard-code numbers are came from NGU paper Appendix H.3.
        # CNN.
        self.conv1 = weight_init(nn.Conv2d(obs_dim[0], 32, kernel_size=8, stride=4))
        self.conv2 = weight_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = weight_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        h = conv2d_out(conv2d_out(conv2d_out(self.obs_dim[1], 4, 4), 4, 2), 3, 1)
        w = conv2d_out(conv2d_out(conv2d_out(self.obs_dim[2], 4, 4), 4, 2), 3, 1)
        # LSTM.
        self.lstm = weight_init(nn.LSTM(h * w, 512))
        self.hx = torch.zeros(self.model_hypr['batch_size'], 512)
        self.cx = torch.zeros(self.model_hypr['batch_size'], 512)
        # Dueling Architecture.
        self.flatten = nn.Flatten()
        self.adv1 = weight_init(nn.Linear(512, 512))
        self.adv2 = weight_init(
            nn.Linear(512, self.act_dim),
            gain=init.calculate_gain('linear'))  # TODO(minho): Convert it into action input
        self.val1 = weight_init(nn.Linear(512, 512))
        self.val2 = weight_init(nn.Linear(512, 1), gain=init.calculate_gain('linear'))

    def forward(self, obs):
        """Dueling network forward pass. Compute Q with V and Adv."""
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        # TODO(minho): action, r_e, r_i, beta
        adv = F.relu(self.adv1(self.hx))
        adv = self.adv2(adv)
        val = F.relu(self.val1(self.hx))
        val = self.val2(val)

        return val + adv - adv.mean(dim=1, keepdim=True)

    def to(self, device):
        super(DuelingLSTM, self).to(device)
        self.hx = self.hx.to(device)
        self.cx = self.cx.to(device)

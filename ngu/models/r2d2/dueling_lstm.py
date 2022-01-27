import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import ngu.utils.pytorch_util as ptu
from ngu.models import model_hypr
from ngu.models.utils import conv2d_out
from ngu.models.common import weight_init
from ngu.models.common.type import Hiddenstate


class DuelingLSTM(nn.Module):
    """Dueling LSTM Model of R2D2"""

    def __init__(self, n_act, obs_shape=(1, 84, 84), model_hypr=model_hypr):
        """Initialize R2D2 Agebt Architecture."""
        super(DuelingLSTM, self).__init__()
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr
        self.N = model_hypr['num_mixtures']  # NGU(N=?) Number.
        # Hard-coded numbers of layers came from NGU paper Appendix H.3.
        # CNN.
        self.conv1 = weight_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4))
        self.conv2 = weight_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = weight_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        h = conv2d_out(conv2d_out(conv2d_out(self.obs_shape[1], 4, 4), 4, 2), 3, 1)
        w = conv2d_out(conv2d_out(conv2d_out(self.obs_shape[2], 4, 4), 4, 2), 3, 1)
        # LSTM.
        self.hidden_units = 512
        self.flatten = nn.Flatten()
        # Output of convolution + action shape + intrinsic reward shape + extrinsic reward shape + num mixture.
        input_shape_lstm = h * w * 64 + (1 + 1 + 1 + self.N)
        self.lstm = weight_init(nn.LSTMCell(input_shape_lstm, 512))
        self.hx = torch.zeros(self.model_hypr['batch_size'], self.hidden_units)
        self.cx = torch.zeros(self.model_hypr['batch_size'], self.hidden_units)
        # Dueling Architecture.
        self.adv1 = weight_init(nn.Linear(self.hidden_units, 512))
        self.adv2 = weight_init(nn.Linear(512, self.n_act), gain=init.calculate_gain('linear'))
        self.val1 = weight_init(nn.Linear(self.hidden_units, 512))
        self.val2 = weight_init(nn.Linear(512, 1), gain=init.calculate_gain('linear'))

    def forward(self, obs, act, int_rew, ext_rew, beta):
        """Dueling network forward pass. Compute Q with V and Adv.

        Args:
            obs: Observation.
            act: Previous action.
            int_rew: Intrinsic `reward.
            ext_rew: Extrinsic reward.
            beta: One-hot vector encoding the value of exploratory factor.
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.cat((x, act, int_rew, ext_rew, beta), dim=1)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        adv = F.relu(self.adv1(self.hx))
        adv = self.adv2(adv)
        val = F.relu(self.val1(self.hx))
        val = self.val2(val)

        return val + adv - adv.mean(dim=1, keepdim=True)

    def to(self, device):
        super(DuelingLSTM, self).to(device)
        self.hx = self.hx.to(device)
        self.cx = self.cx.to(device)

    def get_hidden_state(self, to_cpu=False):
        hx = self.hx.detach()
        cx = self.cx.detach()
        if to_cpu:
            hx = hx.cpu()
            cx = cx.cpu()
        return Hiddenstate(hx, cx)

    def set_hidden_state(self, hidden_state, to_device=True):
        """Set model's hidden state to given hidden state.
        Args:
            hidden_state: Hidden state that model will have.
            to_device: Whether to place it on the device.
        """
        self.hx = hidden_state.hx.clone()
        self.cx = hidden_state.cx.clone()
        if to_device:
            self.hx = self.hx.to(ptu.device)
            self.cx = self.cx.to(ptu.device)

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ngu.models import model_hypr
from ngu.models.utils import conv2d_out
from ngu.models.common import weight_init


class RNDPrediction(nn.Module):
    """RND feature prediction network."""

    def __init__(self, obs_dim=(1, 84, 84), model_hypr=model_hypr):
        super(RNDPrediction, self).__init__()
        self.obs_dim = obs_dim
        self.model_hypr = model_hypr

        # TODO(minho): Take a look at initialization,
        # https://github.com/openai/random-network-distillation/blob/master/utils.py#L35
        # Hard-code numbers are came from NGU paper Appendix H.2.
        # Weight initialization: https://github.com/openai/random-network-distillation/blob/master/policies/cnn_gru_policy_dynamics.py#L125
        self.conv1 = weight_init(nn.Conv2d(obs_dim[0], 32, kernel_size=8, stride=4))
        self.conv2 = weight_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = weight_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        # Linear
        self.flatten = nn.Flatten()
        h = conv2d_out(conv2d_out(conv2d_out(self.obs_dim[1], 4, 4), 4, 2), 3, 1)
        w = conv2d_out(conv2d_out(conv2d_out(self.obs_dim[2], 4, 4), 4, 2), 3, 1)
        self.fc = weight_init(nn.Linear(h * w * 64, 128), init.calculate_gain('linear')) # TODO(minho): Linear gain?

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return self.fc(x)

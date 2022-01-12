import torch
import torch.nn as nn
from torch.nn import init

from ngu.models.utils import conv2d_out
from ngu.models.common import weight_init


class Embedding(nn.Module):
    """Mapping the current observation to a learned representation that the NGU paper refer to as controllable state.
    This can be thought as Siamese network to predict the action taken by the agent to go from one observation to the next.
    """

    def __init__(self, act_dim, obs_dim):
        super(Embedding, self).__init__()
        h = conv2d_out(conv2d_out(conv2d_out(self.obs_dim[1], 4, 4), 4, 2), 3, 1)
        w = conv2d_out(conv2d_out(conv2d_out(self.obs_dim[2], 4, 4), 4, 2), 3, 1)

        # Appendix H.1 for more details about the architecture.
        self.siamese = nn.Sequential(
            weight_init(nn.Conv2d(obs_dim[0], 32, kernel_size=8, stride=4)), nn.ReLU(),
            weight_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            weight_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.Flatten(),
            weight_init(nn.Linear(h * w * 64, 32), init.calculate_gain('linear')), nn.ReLU())

        self.h = nn.Sequential(
            weight_init(nn.Linear(
                32 * 2, 128)),  # 32 for current observation, the other 32 for next observation.
            nn.ReLU(),
            weight_init(nn.Linear(act_dim), init.calculate_gain('linear')),
            nn.Softmax())

    def forward(self, obs_curr, obs_next):
        f_curr = self.siamese(obs_curr)
        f_next = self.siamese(obs_next)
        return self.h(torch.cat((f_curr, f_next), dim=0))

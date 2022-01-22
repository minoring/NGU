import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

from ngu.models.utils import conv2d_out
from ngu.models.common import weight_init


class Embedding(nn.Module):
    """Mapping the current observation to a learned representation that the NGU paper refer to as controllable state.
    This can be thought as Siamese network to predict the action taken by the agent to go from one observation to the next.
    """

    def __init__(self, n_act, obs_shape, model_hypr):
        super(Embedding, self).__init__()
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr

        h = conv2d_out(conv2d_out(conv2d_out(self.obs_shape[1], 4, 4), 4, 2), 3, 1)
        w = conv2d_out(conv2d_out(conv2d_out(self.obs_shape[2], 4, 4), 4, 2), 3, 1)
        # Appendix H.1 for more details about the architecture.
        self.siamese = nn.Sequential(
            weight_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)), nn.ReLU(),
            weight_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            weight_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.Flatten(),
            weight_init(nn.Linear(h * w * 64, 32)), nn.ReLU())
        self.h = nn.Sequential(
            # 32 for current observation, the other 32 for next observation.
            weight_init(nn.Linear(32 * 2, 128)),
            nn.ReLU(),
            weight_init(nn.Linear(128, n_act), init.calculate_gain('linear')),
            nn.Softmax(dim=1))
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.model_hypr['learning_rate_action_prediction'])

    def forward(self, obs_curr, obs_next):
        f_curr = self.siamese(obs_curr)
        f_next = self.siamese(obs_next)
        return self.h(torch.cat((f_curr, f_next), dim=1))

    # TODO(minho): Impl
    def step(self, batch):
        pass

import torch
import torch.nn as nn
import numpy as np

from ngu.models.intrinsic_novelty.lifelong_novelty.rnd_prediction import RNDPrediction
from ngu.models.model_hypr import model_hypr
from ngu.utils.mpi_util import RunningMeanStd


# TODO(minho): Implement normalization for RND
class LifelongNovelty(nn.Module):
    """RND network for life-long novelty bonus.
    It calculate the intrinsic novelty by calculating prediction error between randomly initialized
    neural net (target) and trained one (predictor).
    """

    def __init__(self, obs_dim=(1, 84, 84), model_hypr=model_hypr):
        super(LifelongNovelty, self).__init__()
        self.obs_dim = obs_dim
        self.model_hypr = model_hypr
        self.ll_rms = RunningMeanStd()

        self.predictor = RNDPrediction(obs_dim, model_hypr)
        self.target = RNDPrediction(obs_dim, model_hypr)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        predictor_feature = self.predictor(obs)
        target_feature = self.target(obs)

        return predictor_feature, target_feature

    def compute_lifelong_curiosity(self, obs):
        predictor_feature, target_feature = self(obs)
        err = torch.pow(predictor_feature - target_feature, 2).sum()
        modulator = 1 + (err - self.ll_rms.mean) / np.sqrt(self.ll_rms.var)
        return modulator

    def to(self, device):
        self.predictor.to(device)
        self.target.to(device)

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import ngu.utils.pytorch_util as ptu
from ngu.models.intrinsic_novelty.lifelong_novelty.rnd_prediction import RNDPrediction
from ngu.utils.mpi_util import RunningMeanStd


class LifelongNovelty(nn.Module):
    """RND network for life-long novelty bonus.
    It calculate the intrinsic novelty by calculating prediction error between randomly initialized
    neural net (target) and trained one (predictor).
    """

    def __init__(self, obs_shape, model_hypr, logger):
        super(LifelongNovelty, self).__init__()
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr
        self.logger = logger
        # In order to keep the the rewards on a consistent scale,
        # we normalize the intrinsic reward by dividing it by a running estimate of the
        # standard deviations of the intrinsic returns.
        # Burda et al., 2018.
        self.ll_rms = RunningMeanStd(use_mpi=False)

        self.predictor = RNDPrediction(obs_shape, model_hypr)
        self.target = RNDPrediction(obs_shape, model_hypr)

        # Freeze target network.
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(),
                                    lr=model_hypr['learning_rate_rnd'],
                                    betas=(model_hypr['adam_beta1'], model_hypr['adam_beta2']),
                                    eps=model_hypr['adam_epsilon'])
        self.criterion = nn.MSELoss()
        self.update_count = 0

    def forward(self, obs):
        predictor_feature = self.predictor(obs)
        target_feature = self.target(obs)

        return predictor_feature, target_feature

    @torch.no_grad()
    def compute_lifelong_curiosity(self, obs):
        obs = obs.to(ptu.device)
        predictor_feature, target_feature = self(obs)
        err = ptu.to_numpy(torch.pow(predictor_feature - target_feature, 2)).mean(axis=1)
        self.ll_rms.update(err)  # Update running mean and variance.
        # NGU paper Section 2.
        modulator = 1 + (err - self.ll_rms.mean) / np.sqrt(self.ll_rms.var)
        return torch.from_numpy(modulator)

    def step(self, timestep_seq):
        loss_avg = 0.
        for t in range(len(timestep_seq)):
            pred_f, targ_f = self(timestep_seq[t].state.to(ptu.device))
            loss = self.criterion(targ_f, pred_f)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.predictor.parameters(), self.model_hypr['adam_clip_norm'])
            self.optimizer.step()
            loss_avg = loss_avg + (1 / (t + 1)) * (loss.item() - loss_avg)
        print("RND Loss: {:.4f}".format(loss_avg))

        self.update_count += 1
        self.logger.log_scalar('RNDLoss', loss_avg, self.update_count)
        self.logger.log_scalar('LifelongNoveltyMean', self.ll_rms.mean, self.update_count)
        self.logger.log_scalar('LifelongNoveltyVar', self.ll_rms.var, self.update_count)

    def to(self, device):
        self.predictor.to(device)
        self.target.to(device)

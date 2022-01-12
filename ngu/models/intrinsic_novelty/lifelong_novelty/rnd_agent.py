# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ngu.models.intrinsic_novelty.lifelong_novelty.rnd_intrinsic import RNDIntrinsic
# from ngu.models import model_hypr


# class RNDAgent(nn.Module):
#     """RND agent that the policy is driven by only intrinsic bonus.
#     This model is mainly for experimental purpose.
#     """

#     def __init__(self, act_dim=18, obs_dim=(1, 84, 84), model_hypr=model_hypr):
#         super(RNDAgent, self).__init__()
#         self.act_dim = act_dim
#         self.obs_dim = obs_dim
#         self.model_hypr = model_hypr

#         self.rnd_intrinsic = RNDIntrinsic(obs_dim, model_hypr)
#         self.fc_predictor_act = nn.Linear(128, act_dim)  # Maps feature to action dimension.
#         self.fc_target_act = nn.Linear(128, act_dim)

#         for param in self.fc_target_act.parameters():
#             param.requires_grad = False

#     def forward(self, obs):
#         """Maps feature to action space"""
#         predictor_feature, target_feature = self.rnd_intrinsic(obs)
#         predictor_act = self.fc_predictor_act(F.relu(predictor_feature))
#         target_act = self.fc_target_act(F.relu(target_feature))

#         return predictor_act, target_act

#     @torch.no_grad()
#     def get_action(self, obs):
#         """Select action according to intrinsic reward only, measured by MSE between target and predictor network."""
#         # TODO(minho): Exploration parameter.
#         # TODO(minho): Make sure obs and the model is in same device
#         predictor_act, target_act = self(obs)
#         squared_error = (predictor_act - target_act)**2
#         return squared_error.max(dim=1, keepdim=True).indices

import torch

from ngu.models.intrinsic_novelty.episodic_novelty.embedding import Embedding
from ngu.models import model_hypr
from ngu.utils.mpi_util import RunningMeanStd


class EpisodicNovelty:

    def __init__(self, act_dim, obs_dim, model_hypr=model_hypr):
        """
        Args:
            act_dim: The dimension of environment action, which is the dimension of embedding network.
            model_hypr: Hyperparameters.
        """
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.model_hypr = model_hypr
        self.embedding = Embedding(self.act_dim, self.obs_dim)
        # Running average of the Euclidean distance.
        # This is to make learnt embedding less sensitive to the specific task we are solving.
        self.ed_rms = RunningMeanStd((self.model_hypr['num_neighbors'],))
        self.capacity = model_hypr['episodic_memory_capacity']
        self.episodic_memory = torch.zeros((self.capacity, self.act_dim))
        self.memory_idx = 0

    def clear(self):
        """Clear episodic memory."""
        self.episodic_memory = torch.zeros((self.capacity, self.act_dim))

    def inverse_kernel(self, x, y):
        """Inverse kernel for"""

    def compute_episodic_novelty(self, obs):
        """
        Args:
            obs: Batch of observations that we want to compute episodic novelty.
        """
        # Compute the embedding
        controllable_state = self.embedding(
            obs)  # TODO(minho): How to get obs_curr, obs_next. What if curr_obs is zero?
        # Compute k-nearest neighbor
        euc_dist = ((self.episodic_memory - controllable_state)**2).sum(dim=1).sqrt()
        k_neighbor = euc_dist.topk(self.model_hypr['num_neighbors'])
        # Update the moving average.
        self.ed_rms.update(k_neighbor.values)
        # Normalize Euclidean distance.
        normalized_dist = k_neighbor.values / (self.ed_rms.mean)
        # Cluster the normalized distance, i.e. they become 0 if too small.
        clustered_dist = (normalized_dist - self.model_hypr['cluster_distance'],
                          torch.zeros((self.model_hypr['num_neighbor'], )))
        # Compute the Kernel values between the embedding and its neighbors
        kernel_value = self.model_hypr['kernel_eps'] / (clustered_dist +
                                                        self.model_hypr['kernel_eps'])
        # Compute the similarity between the embedding and its neighbors
        similarity = kernel_value.sum().sqrt() + self.model_hypr['kernel_pseudo_counts_constant']
        if similarity > self.model_hypr['kernel_pseudo_counts_constant']:
            r_intrinsic = 0
        else:
            r_intrinsic = 1 / similarity
        return r_intrinsic

    def insert_state(self, controllable_state):
        """Insert the embedding to the episodic novelty memory."""
        self.episodic_memory[self.memory_idx] = controllable_state
        self.memory_idx = (self.memory_idx + 1) % self.capacity


    def to(self, device):
        self.embedding.to(device)

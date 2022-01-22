import torch

from ngu.models import model_hypr
from ngu.utils.mpi_util import RunningMeanStd
from ngu.models.intrinsic_novelty.episodic_novelty.embedding import Embedding
import ngu.utils.pytorch_util as ptu


class EpisodicNovelty:

    def __init__(self, n_act, obs_shape, model_hypr=model_hypr):
        """
        Args:
            n_act: The dimension of environment action, which is the dimension of embedding network.
            obs_shape: Observation shape.
            model_hypr: Hyperparameters.
        """
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr
        self.embedding = Embedding(self.n_act, self.obs_shape, self.model_hypr)
        # Running average of the Euclidean distance.
        # This is to make learnt embedding less sensitive to the specific task we are solving.
        self.ed_rms = RunningMeanStd((self.model_hypr['num_neighbors'], ))
        self.capacity = model_hypr['episodic_memory_capacity']
        self.episodic_memory = torch.zeros((self.capacity, self.n_act))
        self.memory_idx = 0

    def clear(self):
        """Clear episodic memory."""
        self.episodic_memory = torch.zeros((self.capacity, self.n_act))

    @torch.no_grad()
    def compute_episodic_novelty(self, obs, obs_next):
        """Compute episodic novelty. Take a look at NGU paper Algorithm 1.

        Returns:
            Intrinsic reward of Batch size.
        """
        batch_size = len(obs)
        obs, obs_next = ptu.to_device((obs, obs_next), ptu.device)

        # Compute the embedding
        controllable_state = self.embedding(obs, obs_next)
        controllable_state = controllable_state.detach().cpu()
        # Compute Euclidean distance.
        # Output is BatchSize x MemorySize where element at index (i, j) is distance
        # between ith batch state and jth memory state.
        euc_dist = torch.cdist(controllable_state, self.episodic_memory, p=2.0)
        # Compute k-nearest neighbor
        k_neighbor = euc_dist.topk(self.model_hypr['num_neighbors'], dim=1)
        # Update the moving average.
        self.ed_rms.update(k_neighbor.values)
        # Normalize Euclidean distance.
        normalized_dist = k_neighbor.values / torch.tensor(self.ed_rms.mean).float()
        # Cluster the normalized distance, i.e. they become 0 if too small.
        clustered_dist = torch.max(normalized_dist - self.model_hypr['cluster_distance'],
                                   torch.zeros((
                                       batch_size,
                                       self.model_hypr['num_neighbors'],
                                   )))
        # Compute the Kernel values between the embedding and its neighbors
        kernel_value = self.model_hypr['kernel_epsilon'] / (clustered_dist +
                                                            self.model_hypr['kernel_epsilon'])
        # Compute the similarity between the embedding and its neighbors
        similarity = kernel_value.sum(
            dim=1).sqrt() + self.model_hypr['kernel_pseudo_counts_constant']
        # Only use similarities that is smaller than maximum similarity.
        # If it is bigger than max, use zero.
        r_intrinsic = 1 / similarity
        greater_than_max_idxs = similarity > self.model_hypr['kernel_maximum_similarity']
        r_intrinsic[greater_than_max_idxs] = torch.zeros((1, ))  # Use broadcast to fill in numbers.
        return r_intrinsic

    def insert_state(self, controllable_state):
        """Insert the embedding to the episodic novelty memory."""
        self.episodic_memory[self.memory_idx] = controllable_state
        self.memory_idx = (self.memory_idx + 1) % self.capacity

    def step(self, batch):
        self.embedding.step(batch)

    def to(self, device):
        self.embedding.to(device)

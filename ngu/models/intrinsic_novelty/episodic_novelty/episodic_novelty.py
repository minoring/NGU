import torch

import ngu.utils.pytorch_util as ptu
from ngu.utils.mpi_util import RunningMeanStd
from ngu.models.intrinsic_novelty.episodic_novelty.embedding import Embedding


class EpisodicNovelty:

    def __init__(self, n_actors, n_act, obs_shape, model_hypr, logger):
        """
        Args:
            n_actors: The number of parallel actors.
            n_act: The dimension of environment action, which is the dimension of embedding network.
            obs_shape: Observation shape.
            model_hypr: Hyperparameters.
            logger: Logger.
        """
        self.n_actors = n_actors
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr
        self.logger = logger
        self.controllable_state_dim = 32
        self.embedding = Embedding(self.n_act, self.obs_shape, self.controllable_state_dim,
                                   self.model_hypr, self.logger)
        # Running average of the Euclidean distance.
        # This is to make learnt embedding less sensitive to the specific task we are solving.
        self.ed_rms = RunningMeanStd((self.model_hypr['num_neighbors'], ))
        self.epi_novel_rms = RunningMeanStd() # Running mean standard deviation of episodic novelty.
        self.capacity = model_hypr['episodic_memory_capacity']
        self.episodic_memory = torch.zeros(
            (self.capacity, self.n_actors, self.controllable_state_dim))
        self.memory_idx = 0
        self.update_count = 0

    @torch.no_grad()
    def compute_episodic_novelty(self, obs):
        """Compute episodic novelty. Take a look at NGU paper Algorithm 1.

        Returns:
            Intrinsic reward of Batch size.
        """
        batch_size = len(obs)
        obs = obs.to(ptu.device)
        # Compute the embedding
        controllable_state = self.embedding(obs)
        controllable_state = controllable_state.detach().cpu()
        # Compute Euclidean distance.
        # Output is BatchSize x MemorySize where element at index (i, j) is distance
        # between ith batch state and jth memory state.
        euc_dist = ((self.episodic_memory -
                     controllable_state.unsqueeze(0))**2).sum(dim=-1).sqrt()  # MEM_CAP x BATCH_SIZE
        # Compute k-nearest neighbor
        k_neighbor = euc_dist.topk(self.model_hypr['num_neighbors'], dim=0)
        # Update the moving average.
        self.ed_rms.update(k_neighbor.values.transpose(1, 0))
        # Normalize Euclidean distance.
        normalized_dist = k_neighbor.values / torch.tensor(self.ed_rms.mean).unsqueeze(-1).float()
        # Cluster the normalized distance, i.e. they become 0 if too small.
        clustered_dist = torch.max(normalized_dist - self.model_hypr['cluster_distance'],
                                   torch.zeros((
                                       self.model_hypr['num_neighbors'],
                                       batch_size,
                                   )))
        # Compute the Kernel values between the embedding and its neighbors
        kernel_value = self.model_hypr['kernel_epsilon'] / (clustered_dist +
                                                            self.model_hypr['kernel_epsilon'])
        # Compute the similarity between the embedding and its neighbors
        # N_NEIGHBORS x BATCH_SIZE -> BATCH_SIZE
        similarity = kernel_value.sum(
            dim=0).sqrt() + self.model_hypr['kernel_pseudo_counts_constant']
        # Only use similarities that is smaller than maximum similarity.
        # If it is bigger than max, use zero.
        r_intrinsic = 1 / similarity
        greater_than_max_idxs = similarity > self.model_hypr['kernel_maximum_similarity']
        r_intrinsic[greater_than_max_idxs] = torch.zeros((1, ))  # Use broadcast to fill in numbers.
        # Insert controllable state into episodic memory.
        self.insert_state(controllable_state)

        # Update running mean, std of episodic novelty.
        self.epi_novel_rms.update(r_intrinsic.numpy())

        return r_intrinsic

    def insert_state(self, controllable_state):
        """Insert the embedding to the episodic novelty memory."""
        self.episodic_memory[self.memory_idx] = controllable_state
        self.memory_idx = (self.memory_idx + 1) % self.capacity

    def step(self, timestep_seq):
        self.embedding.step(timestep_seq)

        self.update_count += 1
        self.logger.log_scalar('EpisodicNoveltyMean', self.epi_novel_rms.mean, self.update_count)
        self.logger.log_scalar('EpisodicNoveltyVar', self.epi_novel_rms.mean, self.update_count)
        self.logger.log_scalar('KNeighborEucDistMean', self.ed_rms.mean.mean(), self.update_count)
        self.logger.log_scalar('KNeighborEucDistVar', self.ed_rms.var.mean(), self.update_count)

    def to(self, device):
        self.embedding.to(device)

from collections import deque

import torch
import numpy as np

import ngu.utils.pytorch_util as ptu
from ngu.models.r2d2 import R2D2Learner, R2D2Actor
from ngu.models.r2d2.replay_memory import PrioritizedReplayMemory
from ngu.models.intrinsic_novelty import IntrinsicNovelty
from ngu.models.common.type import Transition, Sequence


class NGUAgent:

    def __init__(self, envs, n_actors, n_act, obs_shape, model_hypr, logger):
        self.envs = envs
        self.n_actors = n_actors
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr
        self.logger = logger
        self.replay_period = self.model_hypr['replay_period']  # Burn-in period.
        # Total sequence length that will be stored in sequence buffer.
        self.seq_len = self.model_hypr['trace_length'] + self.replay_period
        self.n_step = self.model_hypr['n_step']
        # Prioritized Replay Memory to store sequences.
        self.memory = PrioritizedReplayMemory(self.model_hypr['replay_capacity'],
                                              self.model_hypr['replay_priority_exponent'])
        self.r2d2_learner = R2D2Learner(n_act, obs_shape, self.memory, model_hypr)
        self.r2d2_actor = R2D2Actor(envs, n_actors, n_act, obs_shape, self.memory, model_hypr)
        # Initialize Intrinsic Novelty (episodic, life-long module).
        self.intrinsic_novelty = IntrinsicNovelty(n_act, obs_shape, model_hypr)
        # Save previous observation, action, and rewards for r2d2 input.
        self.prev_obs = self.envs.reset()
        self.prev_act = torch.zeros((self.n_actors, 1))
        self.prev_ext_rew = torch.zeros((self.n_actors, 1))  # Previous extrinsic reward.
        self.prev_int_rew = torch.zeros((self.n_actors, 1))  # Previous intrinsic reward.

    def collect_sequence(self):
        """Each Parallel actors collect a sequence for the training.

        Returns:
            [Num actors x total sequence length] shaped tensor.
        """
        fill_steps = self.n_step - 1  # Minimun steps that is needed to fill nstep buffer.
        margin_steps = self.n_step  # Marginal steps that is needed to compute n-step reward.
        num_total_step = self.model_hypr[
            'trace_length'] + self.replay_period + fill_steps + margin_steps
        # Save sequence of Total_Sequence_Length x n_actors transitions.
        seq_buff = []
        init_recurr_state = self.r2d2_actor.policy.get_hidden_state()
        nstep_buff = deque(maxlen=self.n_step)
        for t in range(num_total_step):
            action = self.r2d2_actor.get_eps_greedy_action(self.prev_obs, self.prev_act,
                                                           self.prev_ext_rew, self.prev_int_rew)
            next_obs, rew, done, info = self.envs.step(action)
            # Here done was numpy boolean. Convert it into tensor.
            done = torch.tensor(done[..., np.newaxis])
            self.prev_act, self.prev_ext_rew = action, rew  # Update previous action result.
            self.prev_int_rew = self.intrinsic_novelty.compute_intrinsic_novelty(
                self.prev_obs, next_obs)
            # Fill n-step buffer.
            nstep_buff.append(
                Transition(self.prev_obs, self.prev_act, action, self.prev_int_rew,
                           self.prev_ext_rew, rew, next_obs, done))
            # If we collected n-step transitions, compute n-step reward and store in the
            # sequence buffer.
            if len(nstep_buff) == self.n_step:
                # Compute sum of n-step rewards.
                transition = nstep_buff[0]
                transition.nstep_reward = self.r2d2_actor.compute_nstep_reward(nstep_buff)
                # Store in sequence buffer.
                seq_buff.append(transition)
            self.prev_obs = next_obs
        # After collecting sequences, compute priority.
        priorities = self.compute_priorities(init_recurr_state, seq_buff)
        sequences = self.create_sequence_from_buff(init_recurr_state, seq_buff)
        import pdb
        pdb.set_trace()
        # Store it in replay memory.

    def create_sequence_from_buff(self, init_recurr_state, seq_buff):
        batch_transitions = []
        for actor_idx in range(self.n_actors):
            transitions = []
            for t in range(len(seq_buff)):
                transitions.append(
                    # TODO(minho): Better way?
                    Transition(
                        seq_buff[t].state[actor_idx],
                        seq_buff[t].prev_action[actor_idx],
                        seq_buff[t].action[actor_idx],
                        seq_buff[t].reward_intrinsic[actor_idx],
                        seq_buff[t].reward_extrinsic[actor_idx],
                        seq_buff[t].reward[actor_idx],
                        seq_buff[t].next_state[actor_idx],
                        seq_buff[t].done[actor_idx],
                    ))
            import pdb; pdb.set_trace()
            sequence = Sequence((init_recurr_state[actor_idx:actor_idx + 1][0],
                                 init_recurr_state[actor_idx:actor_idx + 1][1]), transitions,
                                self.r2d2_actor.explr_beta[actor_idx:actor_idx + 1])
            batch_transitions.append(sequence)
        return batch_transitions

    @torch.no_grad()
    def compute_priorities(self, init_recurr_state, sequences):
        """Compute priorities of each actor's sequence.

        Args:
            init_recurr_state: Initial state of recurrent net.
            sequences: Total_Sequence_Length x n_actors transition.
        """
        # Have a burn-in period.
        self.burn_in(init_recurr_state, sequences)
        # Compute TD error.
        td_errors = self.compute_td_error(sequences)
        # Compute priority from TD error.
        eta = self.model_hypr['priority_exponent']
        priorities = eta * td_errors.max(dim=0).values + (1.0 - eta) * td_errors.mean(dim=0)

        return priorities

    def burn_in(self, init_recurr_state, sequences):
        # Set initial hidden states.
        init_hx = init_recurr_state.hx.to(ptu.device)
        init_cx = init_recurr_state.cx.to(ptu.device)
        self.r2d2_actor.policy.set_hidden_state(init_hx, init_cx)

        for t in range(self.replay_period):
            state = sequences[t].state.to(ptu.device)
            prev_act = sequences[t].prev_action.to(ptu.device)
            prev_int_rew = sequences[t].reward_intrinsic.to(ptu.device)
            prev_ext_rew = sequences[t].reward_extrinsic.to(ptu.device)
            beta_onehot = self.r2d2_actor.explr_beta_onehot.to(ptu.device)
            self.r2d2_actor.policy(state, prev_act, prev_int_rew, prev_ext_rew, beta_onehot)
            self.r2d2_actor.target(state, prev_act, prev_int_rew, prev_ext_rew, beta_onehot)

    def compute_td_error(self, sequences):
        """Compute TD-error.

        Args:
            sequences: Total_Sequence_Length x n_actors transition.
        Returns:
            TD-error, which is substraction Q from n-step target.
        """
        td_errors = torch.zeros((self.seq_len - self.replay_period, self.n_actors, 1))
        for t in range(self.replay_period, self.seq_len):
            trans_curr = sequences[t]
            trans_targ = sequences[t + self.n_step]
            # Get current state, action, ... to calcuate Q value.
            obs, prev_act, act, prev_int_rew, prev_ext_rew, nstep_reward, done, beta_onehot = ptu.to_device(
                (trans_curr.state, trans_curr.prev_action, trans_curr.action,
                 trans_curr.reward_intrinsic, trans_curr.reward_extrinsic, trans_curr.nstep_reward,
                 trans_curr.done, self.r2d2_actor.explr_beta_onehot), ptu.device)
            Q = self.r2d2_actor.policy(obs, prev_act, prev_int_rew, prev_ext_rew,
                                       beta_onehot).gather(1, act)
            # Get target state, action, ... to calculate target Q value.
            obs_targ, prev_act_targ, prev_int_rew_targ, prev_ext_rew_targ, done_targ = ptu.to_device(
                (trans_targ.state, trans_targ.prev_action, trans_targ.reward_intrinsic,
                 trans_targ.reward_extrinsic, trans_targ.done), ptu.device)
            targ_Q = self.r2d2_actor.target(obs_targ, prev_act_targ, prev_int_rew_targ,
                                            prev_ext_rew_targ, beta_onehot).max(dim=1,
                                                                                keepdim=True).values
            # TODO(minho): Convert it into transformed Retrace operator.
            # Now, using R2D2 loss.
            h = self.model_hypr['r2d2_reward_transform']
            h_inv = self.model_hypr['r2d2_reward_transformation_inverted']
            gamma = (self.r2d2_actor.discounts**self.n_step).to(ptu.device)
            target_value = h(nstep_reward + (1.0 - torch.logical_and(done, done_targ).float()) *
                             gamma * h_inv(targ_Q))
            td_errors[t - self.replay_period] = target_value - Q
        return td_errors

    def collect_minimum_sequences(self):
        """Agents collect minimun sequence to start the replay."""
        while len(self.memory) < self.model_hypr['minimum_sequences_to_start_replay']:
            self.collect_sequence()
            print("memory size: ", len(self.memory))

    def step(self):
        """Single step update of NGU agent"""
        assert len(self.memory) >= self.model_hypr['minimum_sequences_to_start_replay']
        batch_sequences = self.memory.sample(self.model_hypr['batch_size'])
        self.r2d2_learner.step(batch_sequences)
        self.intrinsic_novelty.step(batch_sequences)

    def to(self, device):
        self.r2d2_learner.to(device)
        self.r2d2_actor.to(device)
        self.intrinsic_novelty.to(device)

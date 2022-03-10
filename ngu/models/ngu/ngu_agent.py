import torch
import numpy as np

import ngu.utils.pytorch_util as ptu
from ngu.models.r2d2 import R2D2Learner, R2D2Actor
from ngu.models.r2d2.replay_memory import PrioritizedReplayMemory
from ngu.models.intrinsic_novelty import IntrinsicNovelty
from ngu.utils import profile
from ngu.utils.mpi_util import RunningMeanStd


class NGUAgent:
    """NEVER GIVE UP!"""

    def __init__(self, envs, n_actors, n_act, obs_shape, model_hypr, logger):
        self.envs = envs
        self.n_actors = n_actors
        self.n_act = n_act
        self.obs_shape = obs_shape
        self.model_hypr = model_hypr
        self.logger = logger
        self.burnin_period = self.model_hypr['burnin_period']  # Burn-in period.
        # Total sequence length that will be stored in sequence buffer.
        self.seq_len = self.model_hypr['trace_length'] + self.burnin_period
        self.n_step = self.model_hypr['n_step']
        self.memory = PrioritizedReplayMemory(self.model_hypr['replay_capacity'], self.seq_len,
                                              self.model_hypr['lstm_hidden_units'])
        # R2D2
        self.r2d2_learner = R2D2Learner(n_act, obs_shape, self.memory, model_hypr, logger)
        self.r2d2_actor = R2D2Actor(envs, n_actors, n_act, obs_shape, model_hypr)
        self.init_hx, self.init_cx = self.r2d2_actor.policy.get_hidden_state(to_cpu=True)
        # Initialize Intrinsic Novelty (episodic, life-long module).
        self.intrinsic_novelty = IntrinsicNovelty(n_actors, n_act, obs_shape, model_hypr, logger)
        # Each actor's current memory index.
        self.local_mem_idx = torch.zeros((self.n_actors, ), dtype=torch.int64)
        # Memory elements.
        self.mem_seq_len = self.seq_len + self.n_step
        self.mem_collected_mask = torch.full((self.n_actors, ), self.mem_seq_len - 1)
        self.state = torch.zeros((self.n_actors, self.mem_seq_len, 1, 84, 84), dtype=torch.float32)
        self.prev_action = torch.zeros((self.n_actors, self.mem_seq_len, 1), dtype=torch.int64)
        self.curr_action = torch.zeros((self.n_actors, self.mem_seq_len, 1), dtype=torch.int64)
        self.intrinsic_reward = torch.zeros((self.n_actors, self.mem_seq_len, 1),
                                            dtype=torch.float32)
        self.extrinsic_reward = torch.zeros((self.n_actors, self.mem_seq_len, 1),
                                            dtype=torch.float32)
        # Save previous observation, action, and rewards for r2d2 input.
        self.prev_obs = self.envs.reset()
        self.prev_act = torch.zeros((self.n_actors, 1), dtype=torch.int64)
        self.prev_ext_rew = torch.zeros((self.n_actors, 1))  # Previous extrinsic reward.
        self.prev_int_rew = torch.zeros((self.n_actors, 1))  # Previous intrinsic reward.

        self.weights_rms = RunningMeanStd()

        self.update_count = 0

    @profile
    @torch.no_grad()
    def collect_sequence(self, random_policy=False):
        """Each Parallel actors collect a sequence and push it into replay memory."""
        # Save transitions of Total_Sequence_Length x n_actors transitions.
        env_steps_collect = 300  # Collect 300 environmental steps.
        for _ in range(env_steps_collect):
            if random_policy:
                action = torch.randint(0, self.n_act, (self.n_actors, 1))
            else:
                action = self.r2d2_actor.get_eps_greedy_action(self.prev_obs, self.prev_act,
                                                               self.prev_int_rew, self.prev_ext_rew)
            next_obs, extr_rew, done, _ = self.envs.step(action)

            # Here done was numpy boolean. Convert it into tensor.
            done = torch.tensor(done)

            # Compute intrinsic rewards.
            intr_rew = self.intrinsic_novelty.compute_intrinsic_novelty(next_obs)

            self.state[:, self.local_mem_idx, ...] = self.prev_obs
            self.prev_action[:, self.local_mem_idx, ...] = self.prev_act
            self.curr_action[:, self.local_mem_idx, ...] = action
            self.intrinsic_reward[:, self.local_mem_idx, ...] = intr_rew
            self.extrinsic_reward[:, self.local_mem_idx, ...] = extr_rew
            # Push to the replay memory if local memory is filled.
            self.push_collected(done)
            self.local_mem_idx = (self.local_mem_idx + 1) % self.mem_seq_len

            # Reset intrinsic memory if it is the end of an episode.
            self.intrinsic_novelty.reset_memory_if_done(done)
            # Reset hidden state if it is the end of an episode.
            self.r2d2_actor.reset_hiddenstate_if_done(done)
            self.prev_int_rew = intr_rew
            self.prev_act = action
            self.prev_ext_rew = extr_rew  # Update previous action result.
            self.prev_obs = next_obs
            self._reset_prev_if_done(done)
            self._reset_local_mem_if_done(done)

    def push_collected(self, done):
        """Push transitions that collected length of `self.mem_seq_len` transitions."""
        collected = self.local_mem_idx == self.mem_collected_mask
        num_collected = collected.sum()
        if num_collected == 0:
            return

        states = self.state[collected]
        prev_acts = self.prev_action[collected]
        curr_acts = self.curr_action[collected]
        intr_rews = self.intrinsic_reward[collected]
        extr_rews = self.extrinsic_reward[collected]
        init_hx = self.init_hx[collected]
        init_cx = self.init_cx[collected]
        done = done[collected]
        intr_factors = self.r2d2_actor.explr_beta[collected]
        intr_factors_onehot = self.r2d2_actor.explr_beta_onehot[collected]
        disc_factors = self.r2d2_actor.discounts[collected]

        amgt_rews = extr_rews + intr_factors.unsqueeze(-1) * intr_rews
        nstep_rews = self.r2d2_actor.compute_nstep_reward(amgt_rews, self.seq_len, disc_factors)

        # Temporarily store hidden states.
        hx, cx = self.r2d2_actor.policy.get_hidden_state()
        # Compute TD error.
        td_errors = self.compute_td_error(num_collected,
                                          self.r2d2_actor,
                                          states,
                                          prev_acts,
                                          curr_acts,
                                          intr_rews,
                                          extr_rews,
                                          amgt_rews,
                                          nstep_rews,
                                          init_hx,
                                          init_cx,
                                          intr_factors_onehot,
                                          disc_factors,
                                          done,
                                          retrace=False)
        priorities = self.compute_priorities(td_errors)
        priorities = ptu.to_list(priorities.squeeze(-1))
        # Recover hidden states.
        self.r2d2_actor.policy.set_hidden_state(hx, cx)

        self.memory.push(states[:, :-self.n_step, ...], prev_acts[:, :-self.n_step, ...],
                         curr_acts[:, :-self.n_step, ...], intr_rews[:, :-self.n_step, ...],
                         extr_rews[:, :-self.n_step], amgt_rews[:, :-self.n_step, ...], nstep_rews,
                         init_hx, init_cx, intr_factors, disc_factors, priorities)

        # Reset memory except for overlapping period.
        overlap = self.model_hypr['overlapping_period']
        self.state[collected, :overlap, ...] = self.state[collected, -overlap:, ...]
        self.prev_action[collected, :overlap, ...] = self.prev_action[collected, -overlap:, ...]
        self.curr_action[collected, :overlap, ...] = self.curr_action[collected, -overlap:, ...]
        self.intrinsic_reward[collected, :overlap, ...] = self.intrinsic_reward[collected,
                                                                                -overlap:, ...]
        self.extrinsic_reward[collected, :overlap, ...] = self.extrinsic_reward[collected,
                                                                                -overlap:, ...]
        self.local_mem_idx[collected] = torch.tensor([overlap], dtype=torch.int64)

    @torch.no_grad()
    def compute_priorities(self, td_errors):
        """Compute priorities of each actor's sequence."""
        # Compute priority from TD error.
        eta = self.model_hypr['r2d2_eta']
        priorities = eta * td_errors.abs().max(dim=0).values + (1.0 -
                                                                eta) * td_errors.abs().mean(dim=0)
        return priorities

    @torch.no_grad()
    def burn_in(self, agent, states, prev_acts, intr_rews, extr_rews, init_hx, init_cx,
                intr_factors):
        """Agent take burn-in period to recover hidden states.
        Timestep x Num batch x (shape) tensors are expected as inputs.
        """
        # Set initial hidden states.
        agent.policy.set_hidden_state(init_hx, init_cx)
        agent.act_sel_net.set_hidden_state(init_hx, init_cx)
        agent.target.set_hidden_state(init_hx, init_cx)

        for t in range(self.burnin_period):
            agent.policy(states[t], prev_acts[t], intr_rews[t], extr_rews[t], intr_factors)
            agent.target(states[t], prev_acts[t], intr_rews[t], extr_rews[t], intr_factors)
        agent.act_sel_net.set_hidden_state(*agent.policy.get_hidden_state())

        for t in range(self.burnin_period, self.seq_len):
            agent.target(states[t], prev_acts[t], intr_rews[t], extr_rews[t], intr_factors)
            agent.act_sel_net(states[t], prev_acts[t], intr_rews[t], extr_rews[t], intr_factors)

    def compute_td_error(self,
                         batch_size,
                         agent,
                         states,
                         prev_acts,
                         curr_acts,
                         intr_rews,
                         extr_rews,
                         amgt_rews,
                         nstep_rews,
                         init_hx,
                         init_cx,
                         intr_factors,
                         disc_factors,
                         done,
                         retrace=True):
        agent.act_sel_net.load_state_dict(agent.policy.state_dict())

        states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews, init_hx, init_cx, intr_factors, disc_factors = self._as_nn_input(
            states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews, init_hx,
            init_cx, intr_factors, disc_factors)

        self.burn_in(agent, states, prev_acts, intr_rews, extr_rews, init_hx, init_cx, intr_factors)

        td_errors = torch.zeros((self.seq_len - self.burnin_period, batch_size, 1),
                                device=ptu.device)
        for t in range(self.seq_len):
            # Get current state, action, ... to calcuate Q value.
            Q = agent.policy(states[t], prev_acts[t], intr_rews[t], prev_acts[t],
                             intr_factors).gather(1, curr_acts[t])
            t_t = t + self.n_step
            next_act = agent.act_sel_net(states[t_t], prev_acts[t_t], intr_rews[t_t],
                                         extr_rews[t_t], intr_factors).argmax(dim=1, keepdim=True)
            targ_Q = agent.target(states[t_t], prev_acts[t_t], intr_rews[t_t], extr_rews[t_t],
                                  intr_factors).gather(1, next_act).detach()

            if retrace:
                # TODO(minho): Implement transformed Retrace operator.
                raise NotImplementedError()
            else:
                # n-step return R2D2 objective.
                h = self.model_hypr['r2d2_reward_transform']
                h_inv = self.model_hypr['r2d2_reward_transformation_inverted']
                gamma = (disc_factors**self.n_step)
                target_value = h(nstep_rews[t] + (1.0 - done.unsqueeze(-1).float().to(ptu.device)) *
                                 gamma * h_inv(targ_Q))
                td_errors[t - self.burnin_period] = target_value - Q

        return td_errors

    def collect_minimum_sequences(self):
        """Agents collect minimun sequence with random policy to start the replay."""
        print("Start collecting minimum sequence to train")
        while len(self.memory) < self.model_hypr['minimum_sequences_to_start_replay']:
            self.collect_sequence(random_policy=True)
            print("Collected {}/{}, {:.1f}%".format(
                len(self.memory), self.model_hypr['minimum_sequences_to_start_replay'],
                (len(self.memory) / self.model_hypr['minimum_sequences_to_start_replay']) * 100))

    @profile
    def step(self):
        """NUM_ACTORS / BATCH_SIZE steps update of NGU agent"""
        assert len(self.memory) >= self.model_hypr['minimum_sequences_to_start_replay']

        num_update_step = max(
            1,
            (self.n_actors // self.model_hypr['batch_size'])) * self.model_hypr['step_per_collect']
        for _ in range(num_update_step):
            transitions, priorities, sequence_idxs = self.memory.sample(
                self.model_hypr['batch_size'])

            states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews, init_hx, init_cx, intr_factors, disc_factors = self._as_nn_input(
                *transitions)
            # Update intrinsic modules.
            self.intrinsic_novelty.step(states[:self.model_hypr['num_frame_intrinsic_train']],
                                        states[1:self.model_hypr['num_frame_intrinsic_traim'] + 1],
                                        curr_acts[:self.model_hypr['num_frame_intrinsic_train']])

            # Update the learner.
            td_errors = self.compute_td_error(self.model_hypr['batch_size'],
                                              self.r2d2_learner,
                                              states,
                                              prev_acts,
                                              curr_acts,
                                              intr_rews,
                                              extr_rews,
                                              amgt_rews,
                                              nstep_rews,
                                              init_hx,
                                              init_cx,
                                              intr_factors,
                                              disc_factors,
                                              retrace=False)
            beta = min(
                1.0, self.model_hypr['beta0'] + (1.0 - self.model_hypr['beta0']) /
                self.model_hypr['beta_decay'] * self.update_count)
            weights = (len(self.memory) * np.array(priorities) / self.memory.total_prios)**(
                -beta)  # Prioritized Experience Replay, Schaul et al., 2016, Algorithm 1.
            self.weights_rms.update(weights)
            weights /= weights.max()
            weights = ptu.to_tensor(weights).unsqueeze(-1)

            self.r2d2_learner.step(td_errors, weights)

            # Update memory priorities with learner.
            new_priorities = self.compute_priorities(td_errors)
            self.memory.update_priorities(sequence_idxs, ptu.to_list(new_priorities.squeeze(-1)))
            self.update_count += 1
            if self.update_count % self.model_hypr['actor_update_period'] == 0:
                print(f"Actors fetch parameters from learner, [learning step: {self.update_count}]")
                self.r2d2_actor.policy.load_state_dict(self.r2d2_learner.policy.state_dict())
                self.r2d2_actor.target.load_state_dict(self.r2d2_actor.policy.state_dict())
            if self.update_count % self.model_hypr['target_q_update_period'] == 0:
                print(f"Updating target Q of learner. [learning step: {self.update_count}]")
                self.r2d2_learner.target.load_state_dict(self.r2d2_learner.policy.state_dict())

            # Log memory size.
            self.logger.log_scalar('R2D2ISWeightMean', self.weights_rms.mean, self.update_count)
            self.logger.log_scalar('R2D2ISWeightVar', self.weights_rms.var, self.update_count)

    @torch.no_grad()
    def get_greedy_action(self, obs, prev_act, prev_ext_rew):
        beta_onehot = self.r2d2_actor.explr_beta_onehot[:1]  # Greey beta.
        prev_int_rew = torch.zeros_like(prev_ext_rew)

        obs, prev_act, prev_ext_rew, prev_int_rew, beta_onehot = ptu.to_device(
            (obs, prev_act, prev_ext_rew, prev_int_rew, beta_onehot), ptu.device)
        greedy_action = self.r2d2_learner.policy(obs, prev_act, prev_ext_rew, prev_int_rew,
                                                 beta_onehot).argmax(1, keepdim=True).cpu()
        return greedy_action

    def init_obs_norm(self):
        """Initializes observation normalization with data from random agent."""
        print("Initializing observation normalization")
        for _ in range(self.model_hypr['init_obs_step']):
            self.envs.step(torch.randint(0, self.n_act, (self.n_actors, 1)))
        self.envs.reset()

    def _as_nn_input(self, states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews,
                     nstep_rews, init_hx, init_cx, intr_factors, disc_factors):
        # NxT -> TxN
        states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews = ptu.transpose_batch(
            [states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews])
        # Send tensors to device.
        states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews, init_hx, init_cx, intr_factors, disc_factors = ptu.to_device(
            [
                states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews, init_hx,
                init_cx, intr_factors, disc_factors
            ], ptu.device)

        return states, prev_acts, curr_acts, intr_rews, extr_rews, amgt_rews, nstep_rews, init_hx, init_cx, intr_factors, disc_factors

    def _reset_prev_if_done(self, done):
        self.prev_obs[done, ...] = torch.zeros(self.obs_shape)
        self.prev_act[done, ...] = torch.zeros((1, ), dtype=torch.int64)
        self.prev_ext_rew[done, ...] = torch.zeros((1, ))
        self.prev_int_rew[done, ...] = torch.zeros((1, ))

    def _reset_local_mem_if_done(self, done):
        self.local_mem_idx[done] = torch.zeros((1, ), dtype=torch.int64)

    def to(self, device):
        self.r2d2_learner.to(device)
        self.r2d2_actor.to(device)
        self.intrinsic_novelty.to(device)

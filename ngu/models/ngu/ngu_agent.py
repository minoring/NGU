from collections import deque

import torch
import numpy as np

import ngu.utils.pytorch_util as ptu
from ngu.models.r2d2 import R2D2Learner, R2D2Actor
from ngu.models.r2d2.replay_memory import PrioritizedReplayMemory
from ngu.models.intrinsic_novelty import IntrinsicNovelty
from ngu.models.common.type import Transition, Sequence, Hiddenstate
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
        self.replay_period = self.model_hypr['replay_period']  # Burn-in period.
        # Total sequence length that will be stored in sequence buffer.
        self.seq_len = self.model_hypr['trace_length'] + self.replay_period
        self.n_step = self.model_hypr['n_step']
        fill_steps = self.n_step - 1  # Minimun steps that is needed to fill nstep buffer.
        margin_steps = self.n_step  # Marginal steps that is needed to compute n-step reward.
        self.num_total_step = self.model_hypr[
            'trace_length'] + self.replay_period + fill_steps + margin_steps
        self.memory = PrioritizedReplayMemory(self.model_hypr['replay_capacity'])
        # R2D2
        self.r2d2_learner = R2D2Learner(n_act, obs_shape, self.memory, model_hypr, logger)
        self.r2d2_actor = R2D2Actor(envs, n_actors, n_act, obs_shape, model_hypr)
        # Initialize Intrinsic Novelty (episodic, life-long module).
        self.intrinsic_novelty = IntrinsicNovelty(n_actors, n_act, obs_shape, model_hypr, logger)
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
        """Each Parallel actors collect a sequence for the training.

        Returns:
            [Num actors x total sequence length] shaped tensor.
        """
        # Save transitions of Total_Sequence_Length x n_actors transitions.
        transitions = []
        init_recurr_state = self.r2d2_actor.policy.get_hidden_state()
        intrinsic_factor = self.r2d2_actor.explr_beta_onehot
        discount_factor = self.r2d2_actor.discounts
        nstep_buff = deque(maxlen=self.n_step)
        actor_done_mask = torch.zeros((self.n_actors, 1))
        # Collect NUM_TOTAL_STEP transitions.
        for _ in range(self.num_total_step):
            if random_policy:
                action = torch.randint(0, self.n_act, (self.n_actors, 1))
            else:
                action = self.r2d2_actor.get_eps_greedy_action(self.prev_obs, self.prev_act,
                                                               self.prev_int_rew, self.prev_ext_rew)
            next_obs, rew, done, info = self.envs.step(action)

            # Here done was numpy boolean. Convert it into tensor.
            done = torch.tensor(done[..., np.newaxis])
            actor_done_mask = torch.logical_or(done, actor_done_mask)
            # Reset memory if it is the end of an episode.
            self.intrinsic_novelty.reset_memory_if_done(done)
            # Reset hidden state if it is the end of an episode.
            self.r2d2_actor.reset_hiddenstate_if_done(done)
            self.prev_act, self.prev_ext_rew = action, rew  # Update previous action result.
            # Fill n-step buffer.
            intrinsic_novelty = self.intrinsic_novelty.compute_intrinsic_novelty(next_obs)
            reward_augmented = rew + self.r2d2_actor.explr_beta * intrinsic_novelty
            nstep_buff.append(
                Transition(self.prev_obs, self.prev_act, action, self.prev_int_rew,
                           self.prev_ext_rew, reward_augmented, next_obs, done))
            self.prev_int_rew = intrinsic_novelty

            self._reset_prev_if_done(done)
            # If we collected n-step transitions, compute n-step reward and store in the
            # sequence buffer.
            if len(nstep_buff) == self.n_step:
                # Compute sum of n-step rewards.
                transition = nstep_buff[0]
                nstep_reward = self.r2d2_actor.compute_nstep_reward(nstep_buff,
                                                                    self.r2d2_actor.discounts)
                # Store in sequence buffer.
                transitions.append(transition._replace(nstep_reward=nstep_reward))
                # SEQ_LENGTH x NUM_ACTOR
            self.prev_obs = next_obs
        # After collecting sequences, compute priority.
        timestep_seq = Sequence(init_recurr_state, transitions, intrinsic_factor, discount_factor)
        # Temporarily store hidden states.
        policy_hidden_state = self.r2d2_actor.policy.get_hidden_state()
        # Compute TD error.
        td_errors = self.compute_td_error(self.n_actors, self.r2d2_actor, timestep_seq)
        priorities = self.compute_priorities(td_errors)
        priorities = ptu.to_list(priorities.squeeze(-1))
        # L x STATE x NUM_ACTOR -> NUM_ACTOR x L x STATE -> L x STATE x NUM_ACTOR
        sequences = self.batch_seq_from_timestep_seq(timestep_seq)
        # Putting in prioritized replay memory.
        self.memory.push(sequences, priorities, actor_done_mask)

        # Recover hidden states.
        self.r2d2_actor.policy.set_hidden_state(policy_hidden_state)

    @profile
    def batch_seq_from_timestep_seq(self, timestep_seq):
        """Given timestep sequences, create batch of sequence.

        Args:
            timestep_seq: Sequence(init_recurr_state=[N_ACTORS, 1],
                            transitions=[SEQUENCE_LENGTH, N_ACTORS] transitions,
                            intrinsic_factor=[N_ACTORS, 1],
                            discount_factor=[N_ACTORS, 1]) shaped sequence.
        Returns:
            (N_ACTORS, SEQUENCE) shaped list.
        """
        sequences = []
        transitions = timestep_seq.transitions
        init_recurr_state = timestep_seq.init_recurr_state
        intrinsic_factor = timestep_seq.intrinsic_factor
        discount_factor = timestep_seq.discount_factor
        # TODO(minho): Better way?
        for actor_idx in range(self.n_actors):
            actor_transitions = []
            for t in range(len(transitions)):
                actor_transitions.append(
                    Transition(
                        transitions[t].state[actor_idx],
                        transitions[t].prev_action[actor_idx],
                        transitions[t].action[actor_idx],
                        transitions[t].reward_intrinsic[actor_idx],
                        transitions[t].reward_extrinsic[actor_idx],
                        transitions[t].reward_augmented[actor_idx],
                        transitions[t].next_state[actor_idx],
                        transitions[t].done[actor_idx],
                        transitions[t].nstep_reward[actor_idx],
                    ))
            sequence = Sequence(
                Hiddenstate(hx=init_recurr_state.hx[actor_idx:actor_idx + 1].cpu(),
                            cx=init_recurr_state.cx[actor_idx:actor_idx + 1].cpu()),
                actor_transitions, intrinsic_factor[actor_idx:actor_idx + 1],
                discount_factor[actor_idx:actor_idx + 1])
            sequences.append(sequence)
        return sequences

    @torch.no_grad()
    def compute_priorities(self, td_errors):
        """Compute priorities of each actor's sequence."""
        # Compute priority from TD error.
        eta = self.model_hypr['r2d2_eta']
        priorities = eta * td_errors.abs().max(dim=0).values + (1.0 -
                                                                eta) * td_errors.abs().mean(dim=0)
        return priorities

    @torch.no_grad()
    def burn_in(self, agent, timestep_seq):
        """Learner take burn-in period to recover hidden states."""
        # Set initial hidden states.
        agent.policy.set_hidden_state(timestep_seq.init_recurr_state)
        agent.act_sel_net.set_hidden_state(timestep_seq.init_recurr_state)
        agent.target.set_hidden_state(timestep_seq.init_recurr_state)
        beta_onehot = timestep_seq.intrinsic_factor.to(ptu.device)

        transitions = timestep_seq.transitions
        for t in range(self.replay_period):
            state = transitions[t].state.to(ptu.device)
            prev_act = transitions[t].prev_action.to(ptu.device)
            prev_int_rew = transitions[t].reward_intrinsic.to(ptu.device)
            prev_ext_rew = transitions[t].reward_extrinsic.to(ptu.device)
            agent.policy(state, prev_act, prev_int_rew, prev_ext_rew, beta_onehot)
            agent.target(state, prev_act, prev_int_rew, prev_ext_rew, beta_onehot)
        agent.act_sel_net.set_hidden_state(agent.policy.get_hidden_state())

        for t in range(self.replay_period, len(timestep_seq)):
            state = transitions[t].state.to(ptu.device)
            prev_act = transitions[t].prev_action.to(ptu.device)
            prev_int_rew = transitions[t].reward_intrinsic.to(ptu.device)
            prev_ext_rew = transitions[t].reward_extrinsic.to(ptu.device)
            agent.target(state, prev_act, prev_int_rew, prev_ext_rew)
            agent.act_sel_net(state, prev_act, prev_int_rew, prev_ext_rew)

    def compute_td_error(self, batch_size, agent, timestep_seq):
        """Compute TD-error.

        Args:
            batch_size: Batch size.
            agent: R2D2 Actor or Learner.
            timestep_seq: Sequence(init_recurr_state=[N_ACTORS, 1],
                                   transitions=[SEQUENCE_LENGTH, N_ACTORS] transitions,
                                   intrinsic_factor=[N_ACTORS, 1],
                                   discount_factor=[N_ACTORS, 1]) shaped sequence.
        Returns:
            TD-error, which is substraction Q from n-step target.
        """
        agent.act_sel_net.load_state_dict(agent.policy.state_dict())

        self.burn_in(agent, timestep_seq)

        td_errors = torch.zeros((self.seq_len - self.replay_period, batch_size, 1),
                                device=ptu.device)
        for t in range(self.replay_period, self.seq_len):
            trans_curr = timestep_seq.transitions[t]
            trans_targ = timestep_seq.transitions[t + self.n_step]

            # Get current state, action, ... to calcuate Q value.
            obs, prev_act, act, prev_int_rew, prev_ext_rew, nstep_reward, done, beta_onehot = ptu.to_device(
                (trans_curr.state, trans_curr.prev_action, trans_curr.action,
                 trans_curr.reward_intrinsic, trans_curr.reward_extrinsic, trans_curr.nstep_reward,
                 trans_curr.done, timestep_seq.intrinsic_factor), ptu.device)
            Q = agent.policy(obs, prev_act, prev_int_rew, prev_ext_rew, beta_onehot).gather(1, act)

            # Get target state, action, ... to calculate target Q value.
            obs_targ, prev_act_targ, prev_int_rew_targ, prev_ext_rew_targ, done_targ = ptu.to_device(
                (trans_targ.state, trans_targ.prev_action, trans_targ.reward_intrinsic,
                 trans_targ.reward_extrinsic, trans_targ.done), ptu.device)

            next_act = agent.act_sel_net(obs_targ, prev_act_targ, prev_int_rew_targ,
                                         prev_ext_rew_targ, beta_onehot).argmax(dim=1, keepdim=True)
            targ_Q = agent.target(obs_targ, prev_act_targ, prev_int_rew_targ, prev_ext_rew_targ,
                                  beta_onehot).gather(1, next_act).detach()

            # TODO(minho): Convert it into transformed Retrace operator.
            # Now, using R2D2 loss.
            h = self.model_hypr['r2d2_reward_transform']
            h_inv = self.model_hypr['r2d2_reward_transformation_inverted']
            gamma = (timestep_seq.discount_factor**self.n_step).to(ptu.device)

            target_value = h(nstep_reward + (1.0 - torch.logical_and(done, done_targ).float()) *
                             gamma * h_inv(targ_Q))
            td_errors[t - self.replay_period] = target_value - Q

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
            sequences, priorities, sequence_idxs = self.memory.sample(self.model_hypr['batch_size'])
            timestep_seq = self._batch_seq_to_timestep_seq(sequences)

            # Update intrinsic modules.
            self.intrinsic_novelty.step(
                timestep_seq.transitions[:self.model_hypr['num_frame_intrinsic_train']])

            # Update the learner.
            td_errors = self.compute_td_error(self.model_hypr['batch_size'], self.r2d2_learner,
                                              timestep_seq)
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
            self.logger.log_scalar('MemorySize', len(self.memory), self.update_count)
            self.logger.log_scalar('R2D2ISWeightMean', self.weights_rms.mean, self.update_count)
            self.logger.log_scalar('R2D2ISWeightVar', self.weights_rms.var, self.update_count)

            # Every n learning steps, any excess data about the memory capacity threshold is removed in FIFO order.
            if self.update_count % self.model_hypr['remove_to_fit_interval'] == 0:
                print(f"Removing memory to fit capacity. [learning step: {self.update_count}]")
                print(f"Before the removal, memory size: {len(self.memory)}")
                self.memory.remove_to_fit()
                print(f"After the removal, memory size: {len(self.memory)}")

    def _batch_seq_to_timestep_seq(self, sequences):
        """Convert batch of sequence that is sampled from the memory to the format that the training loop expects (timestep_seq).
        It is bit ugly but works...
        """
        # [NUM_ACTOR x [Transition, ...] <- 120 ] Transition(state, action, next_obs)...
        # [120 x Transition.state = [NUM_ACTORS state], Transition.action = [NUM_ACTORS]]

        # Convert NUM_ACTORS sequence into SEQUENCE_LENGTH x NUM_ACTORS sequence.
        # Example that could give you intuitions to understand the code.
        # **************************
        # a = Sequence(init_recurr_state=[1], transitions=[1, 2, 3], intrinsic_factor=[1])
        # b = Sequence(init_recurr_state=[2], transitions=[4, 5, 6], intrinsic_factor=[2])
        # c = Sequence(init_recurr_state=[3], transitions=[7, 8, 9], intrinsic_factor=[3])
        # sequences = [a, b, c]
        # batch = Sequence(*zip(*sequences))
        # batch2 = Sequence(list(zip(*(batch.init_recurr_state))), list(zip(*(batch.transitions))),
        #                 list(zip(*(batch.intrinsic_factor))))
        # batch =====> Sequence(init_recurr_state=([1], [2], [3]), transitions=([1, 2, 3], [4, 5, 6], [7, 8, 9]), intrinsic_factor=([1], [2], [3]))
        # batch2 ====> Sequence(init_recurr_state=[(1, 2, 3)], transitions=[(1, 4, 7), (2, 5, 8), (3, 6, 9)], intrinsic_factor=[(1, 2, 3)])
        # **************************
        # Sample from memory, convert it into what training code expect.
        field_assembled = Sequence(*zip(*sequences))
        timestep_seq = Sequence(list(zip(*field_assembled.init_recurr_state)),
                                list(zip(*field_assembled.transitions)),
                                torch.cat(field_assembled.intrinsic_factor),
                                torch.cat(field_assembled.discount_factor))
        timestep_seq = timestep_seq._replace(
            init_recurr_state=Hiddenstate(torch.cat(timestep_seq.init_recurr_state[0]),
                                          torch.cat(timestep_seq.init_recurr_state[1])))
        timestep_seq = timestep_seq._replace(
            transitions=[Transition(*zip(*trans)) for trans in timestep_seq.transitions])
        timestep_seq = timestep_seq._replace(transitions=[
            Transition(*(torch.stack(item) for item in lst)) for lst in timestep_seq.transitions
        ])

        return timestep_seq

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
        for s in range(self.model_hypr['init_obs_step']):
            self.envs.step(torch.randint(0, self.n_act, (self.n_actors, 1)))
        self.envs.reset()

    def _reset_prev_if_done(self, done):
        self.prev_obs[done.squeeze(-1), :] = torch.zeros(self.obs_shape)
        self.prev_act[done.squeeze(-1), :] = torch.zeros((1, ), dtype=torch.int64)
        self.prev_ext_rew[done.squeeze(-1), :] = torch.zeros((1, ))
        self.prev_int_rew[done.squeeze(-1), :] = torch.zeros((1, ))

    def to(self, device):
        self.r2d2_learner.to(device)
        self.r2d2_actor.to(device)
        self.intrinsic_novelty.to(device)

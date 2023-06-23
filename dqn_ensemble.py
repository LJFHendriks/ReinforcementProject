from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import DQN

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import DQNPolicy

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class DQNEnsemble(DQN):

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        temperature: float = 10
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
        )
        self.temperature = temperature

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)

                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=-1)

                # 1-step TD target
                rewards = replay_data.rewards.t().expand(self.policy.ensemble_size, -1)
                dones = replay_data.dones.t().expand(self.policy.ensemble_size, -1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                std = next_q_values.std(0) if self.policy.ensemble_size > 1 else th.zeros(next_q_values.shape[-1])
                target_weight = th.sigmoid(-std * self.temperature) + 0.5

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            action_index = replay_data.actions.expand(self.policy.ensemble_size, -1, -1).long()
            current_q_values = th.gather(current_q_values, dim=-1, index=action_index).squeeze(-1)
            bellman = th.pow(current_q_values - target_q_values, 2)

            # Compute weighted Bellman backup loss
            loss_tensor = target_weight * bellman

            #Compute mean over batches per model
            try:
                loss = th.mean(replay_data.mask.t() * loss_tensor, dim=1)
            except AttributeError:
                loss = th.mean(loss_tensor, dim=1)
            losses.append(loss.mean().item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.sum().backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
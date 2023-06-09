from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Any, Dict, List, Union, Optional, NamedTuple

from gym import spaces
import torch as th
import numpy as np

from stable_baselines3.common.vec_env import VecNormalize


class EnsembleReplayBuffer(ReplayBuffer):
    """Replay buffer that stores a mask for ensemble learning"""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        ensemble_size: int = 1,
        beta: float = 1.
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination
        )
        self.ensemble_size = ensemble_size
        self.beta = beta
        self.mask = np.zeros((self.buffer_size, self.ensemble_size), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        self.mask[self.pos] = np.array(th.full((self.ensemble_size,), self.beta).bernoulli()).copy()

        super().add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos
        )

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        samples = super()._get_samples(batch_inds, env)

        return EnsembleReplayBufferSamples(*samples, self.to_torch(self.mask[batch_inds, :]))


class EnsembleReplayBufferSamples(NamedTuple, ReplayBufferSamples):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    mask: th.Tensor

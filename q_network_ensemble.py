import random
from typing import List, Optional, Type

from gym import spaces
import torch as th
from torch import nn, vmap

from torch.func import stack_module_state, functional_call
import copy

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp
)

from stable_baselines3.dqn.policies import QNetwork


class QNetworkEnsemble(QNetwork):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            ensemble_size: int = 1,
            l: float = 1
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images
        )
        action_dim = self.action_space.n
        modules = [self.q_net] + [nn.Sequential(*create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)) for _ in range(1, ensemble_size)]
        self.q_net = Ensemble(modules)
        self.ensemble_size = ensemble_size
        self.l = l

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)

        # UCB exploration
        result = q_values.mean(0)
        if self.ensemble_size > 1:
            result += self.l * q_values.std(0)
        action = result.argmax(-1)
        return action


class Ensemble(nn.Module):
    def __init__(self, models, **kwargs):
        super().__init__()

        self.models = nn.ModuleList(models)

        # base_model = copy.deepcopy(models[0])
        # base_model = base_model.to('meta')
        #
        # def fmodel(params, buffers, x):
        #     return functional_call(base_model, (params, buffers), (x,))
        #
        # self.vmap_model = vmap(fmodel, in_dims=(0, 0, None))

    def forward(self, *args, **kwargs):
        # params, buffers = stack_module_state(self.models)
        #
        # return self.vmap_model(params, buffers, *args, **kwargs)
        return th.stack([model(args[0]) for model in self.models])

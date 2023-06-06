from typing import List, Optional, Type

from gym import spaces
from torch import nn, vmap

from torch.func import stack_module_state, functional_call
import copy

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp
)

from stable_baselines3.dqn.policies import QNetwork


class EnsembleQNetwork(QNetwork):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            ensemble_size: int = 1
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
        modules = [nn.Sequential(*create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)) for _ in range(ensemble_size)]
        self.q_net = Ensemble(modules)


class Ensemble(nn.Module):
    def __init__(self, modules, **kwargs):
        super().__init__()

        self.params, self.buffers = stack_module_state(modules)

        base_model = copy.deepcopy(modules[0])
        base_model = base_model.to('meta')

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))

        self.vmap_model = vmap(fmodel, in_dims=(0, 0, None))

        for key, param in self.params.items():
            self.register_parameter(key.replace('.', '_'), nn.Parameter(param))

        for key, buffer in self.buffers.items():
            self.register_buffer(key, buffer)

    def forward(self, *args, **kwargs):
        return self.vmap_model(self.params, self.buffers, *args, **kwargs)

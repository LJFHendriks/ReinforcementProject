from typing import TypeVar, Literal

from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import MaybeCallback

SelfDQN = TypeVar("SelfDQN", bound="DQN")


def reset_parameters(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


class DQNResetting(DQN):
    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
        resets: int = 0
    ) -> SelfDQN:
        init_optimizer_state = self.policy.optimizer.state_dict()
        result = self
        for i in range(resets + 1):
            result = super().learn(
                total_timesteps=int(total_timesteps/(resets+1)),
                callback=callback,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )
            if i < resets:
                # Reset the weights of the model to random values
                self.policy.q_net.q_net.apply(reset_parameters)
                self.policy.q_net_target.q_net.apply(reset_parameters)
                self.policy.optimizer.load_state_dict(init_optimizer_state)

        return result

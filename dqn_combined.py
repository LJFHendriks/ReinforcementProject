from typing import TypeVar, Literal

from gym import spaces
from stable_baselines3.common.type_aliases import MaybeCallback

from dqn_ensemble import DQNEnsemble
from dqn_resetting import DQNResetting, reset_parameters
from itertools import product

SelfDQN = TypeVar("SelfDQN", bound="DQN")

ResetTypes = Literal["concurrent", "sequential"]


class DQNCombined(DQNEnsemble, DQNResetting):
    def learn(
            self: SelfDQN,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "DQN",
            reset_num_timesteps: bool = False,
            progress_bar: bool = False,
            resets: int = 0,
            reset_mode: ResetTypes = "concurrent"
    ) -> SelfDQN:
        if reset_mode == "concurrent":
            return super().learn(
                total_timesteps=int(total_timesteps / (resets + 1)),
                callback=callback,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
                resets=resets
            )
        elif reset_mode == "sequential":
            total_resets = resets*self.policy.ensemble_size
            for reset, model in product(range(resets), range(self.policy.ensemble_size)):
                super().learn(
                    total_timesteps=int(total_timesteps / (total_resets + 1)),
                    callback=callback,
                    log_interval=log_interval,
                    tb_log_name=tb_log_name,
                    reset_num_timesteps=reset_num_timesteps,
                    progress_bar=progress_bar,
                    resets=0
                )
                self.policy.q_net.q_net.models[model].apply(reset_parameters)
                self.policy.q_net_target.q_net.models[model].apply(reset_parameters)
            return super().learn(
                total_timesteps=int(total_timesteps / (total_resets + 1)),
                callback=callback,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
                resets=0
            )
        else:
            raise ValueError("Invalid reset mode. Expected one of: %s" % ResetTypes)


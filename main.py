import gym

from typing import Callable
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv


hyperparams = {
    "learning_rate": 6.3e-4,
    "batch_size": 128,
    "buffer_size": 50000,
    "learning_starts": 0,
    "gamma": 0.99,
    "target_update_interval": 250,
    "train_freq": 4,
    "gradient_steps": -1,
    "exploration_fraction": 0.12,
    "exploration_final_eps": 0.1,
    "policy_kwargs": dict(net_arch=[256, 256])
}


def make_env(env_id: str, rank: int, seed: int=0) -> Callable:
    def _init() -> gym.Env:
        environ = gym.make(env_id)
        environ.reset()
        return environ

    set_random_seed(seed)
    return _init


def run_experiment():
    env_id = "LunarLander-v2"
    num_cpu = 4

    # Wrap the environment with a Monitor to log the results
    log_dir = "/scratch/cjwever/AIDM_Convergence_1/logs/"
    env = SubprocVecEnv([lambda: Monitor(gym.make(env_id), log_dir) for _ in range(num_cpu)])

    model = DQN("MlpPolicy", env, **hyperparams, verbose=1)

    # Train the agent
    model.learn(total_timesteps=int(5e6),  progress_bar=True)

    # Save the agent
    model.save("dqn_lunar")
    del model  # delete trained model to demonstrate loading


if __name__ == '__main__':
    run_experiment()
import sys

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN

from dqn_ensemble import DQNEnsemble
from ensemble_policy import EnsemblePolicy

# log_location = sys.argv[1]


hyperparams = {
    "learning_rate": 6.3e-4,
    "batch_size": 1280,
    "buffer_size": 1000000,
    "learning_starts": 0,
    "gamma": 0.9,
    "target_update_interval": 500,
    "train_freq": 4,
    "gradient_steps": -1,
    "exploration_fraction": 0.12,
    "exploration_final_eps": 0.1,
    "policy_kwargs": dict(net_arch=[256, 256], ensemble_size=5)
}

# Create environment
env = gym.make("LunarLander-v2")

# Wrap the environment with a Monitor to log the results
log_dir = "/scratch/cjwever/"
env = Monitor(env, log_dir)

# Instantiate the agent
model = DQNEnsemble(EnsemblePolicy, env, **hyperparams, verbose=1)

# Train the agent
model.learn(total_timesteps=int(5e6),  progress_bar=False)

# Save the agent
model.save(log_dir + "dqn_lunar")
del model  # delete trained model to demonstrate loading

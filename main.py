import sys
import os

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN

from dqn_ensemble import DQNEnsemble
from ensemble_policy import EnsemblePolicy
from ensemble_replay_buffer import EnsembleReplayBuffer

try:
    log_dir = sys.argv[1]
except IndexError:
    log_dir = "result_log"

ensemble_size = 5

hyperparams = {
    "replay_buffer_class": EnsembleReplayBuffer,
    "replay_buffer_kwargs": dict(ensemble_size=ensemble_size),
    "policy_kwargs": dict(ensemble_size=ensemble_size)
}

# Create environment
env = gym.make("LunarLander-v2")

# Wrap the environment with a Monitor to log the results
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Instantiate the agent
model = DQNEnsemble(EnsemblePolicy, env, **hyperparams, verbose=1)

# Train the agent
model.learn(total_timesteps=int(5e6),  progress_bar=False)

# Save the agent
model.save(log_dir + "dqn_lunar")
del model  # delete trained model to demonstrate loading

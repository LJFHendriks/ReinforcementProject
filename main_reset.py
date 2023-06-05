import sys
import os

import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN

log_location = sys.argv[1]

resets = int(sys.argv[2])

timesteps = int(5e6)

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
    "policy_kwargs": dict(net_arch=[256, 256])
}

# Create environment
env = gym.make("LunarLander-v2")

# Wrap the environment with a Monitor to log the results
log_dir = "/scratch/cjwever/" + log_location
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)

# Reset the parameters of all the layers
def reset_parameters(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()

init_optimizer_state=model.policy.optimizer.state_dict()
# Train the agent
resets=resets+1 # since the loop ends with an update
for i in range(0,resets):
    # Train for one step
    model.learn(total_timesteps=int(timesteps/resets), log_interval=4,reset_num_timesteps=False)
    if i != resets:
        # Reset the weights of the model to random values
        model.policy.q_net.q_net.apply(reset_parameters)
        model.policy.q_net_target.q_net.apply(reset_parameters)
        model.policy.optimizer.load_state_dict(init_optimizer_state)

# Save the agent
model.save(log_dir + "dqn_lunar")
del model  # delete trained model to demonstrate loading

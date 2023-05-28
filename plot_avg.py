
import numpy as np
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from glob import glob

log_dirs = sys.argv[1]
monitor_files = glob(os.path.join(log_dirs, "*"))

def plot_average_curve(log_dirs, title="Average Learning Curve"):
  ys=[]
  for dir in log_dirs:
    results = load_results(dir)
    r_list = results['r']
    l_list = results['l']
    y=[]
    for r, l in zip(r_list, l_list):
        for i in range(l):
          y.append(r)
    y = moving_average(y, window=10000)
    ys.append(y)

  min_len = np.inf
  for y in ys:
    min_len = min(min_len, len(y))
  for i in range(len(ys)):
    ys[i] = ys[i][:min_len]

  x= range(min_len)
  # Compute average and standard deviation of y lists
  avg_rewards = np.mean(ys, axis=0)
  std_rewards = np.std(ys, axis=0)

  # Plot average reward and standard deviation
  fig = plt.figure(title)
  plt.plot(x, avg_rewards)
  plt.fill_between(x, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.5)
  plt.xlabel("Number of Timesteps")
  plt.ylabel("Rewards")
  plt.title(title + " Smoothed")
  plt.ylim(-800, 300)
  plt.show()

plot_average_curve(monitor_files)
import numpy as np
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from glob import glob
import sys

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_multiple_average_curve(log_dirs_list, title="Average Learning Curve"):
  yss=[]
  all_ys=[]
  for log_dirs in log_dirs_list:
    ys=[]
    for dir in log_dirs:
      results = load_results(dir)
      r_list = results['r']
      l_list = results['l']
      y=[]
      for r, l in zip(r_list, l_list):
          for i in range(l):
            y.append(r)
      print(len(y))
      y = moving_average(y, window=20000)
      ys.append(y)
      all_ys.append(y)
    yss.append(ys)

  min_len = np.inf
  for y in all_ys:
    min_len = min(min_len, len(y))
  avg_rewards = []
  std_rewards = []
  for i in range(len(yss)):
    for j in range(len(yss[i])):
      yss[i][j] = yss[i][j][:min_len]
    avg_rewards.append(np.mean(yss[i], axis=0))
    std_rewards.append(np.std(yss[i], axis=0))
    

  x = range(min_len)

  # Plot average reward and standard deviation
  fig = plt.figure(title)
  for i in range(len(avg_rewards)):
    plt.plot(x, avg_rewards[i])
    plt.fill_between(x, avg_rewards[i] - std_rewards[i], avg_rewards[i] + std_rewards[i], alpha=0.5)
  plt.xlabel("Number of Timesteps")
  plt.ylabel("Rewards")
  plt.title(title + " Smoothed")
  plt.ylim(-800, 300)
  plt.show()


log_dirs1 = sys.argv[1]
log_dirs2 = sys.argv[2]
monitor_files1 = glob(os.path.join(log_dirs1, "*"))
monitor_files2 = glob(os.path.join(log_dirs2, "*"))
plot_multiple_average_curve([monitor_files1, monitor_files2])
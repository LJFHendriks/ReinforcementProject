import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
import argparse
from plot_average import moving_average


def plot_multiple_average_curve(dirs, labels, title=None):
    assert len(dirs) == len(labels), 'The number of labels provides should be equal to the log dirs'
    title = "Average Learning Curve" if title is None else title
    dirs = [[os.path.join(dir, subdir) for subdir in os.listdir(dir)] for dir in dirs]
    yss=[]
    all_ys=[]
    for log_dirs in dirs:
        ys=[]
        for dir in log_dirs:
          results = load_results(dir)
          y = np.repeat(results.r, results.l)
          print(len(y))
          y = moving_average(y, window=40000)
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
    lines = []
    for i in range(len(dirs)):
        label = labels[i]
        plt.plot(x, avg_rewards[i],label=label)
        plt.fill_between(x, avg_rewards[i] - std_rewards[i], avg_rewards[i] + std_rewards[i], alpha=0.5)


    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.ylim(-200, 300)
    plt.legend(loc="upper left")
    plt.savefig(f'plots/{title}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot multiple average curves')
    parser.add_argument('dirs', type=str, nargs='+')
    parser.add_argument('-l', '--labels', type=str, nargs='*')
    parser.add_argument('-t', '--title', type=str)

    args = parser.parse_args()
    if args.labels is None or len(args.labels) == 0:
        args.labels = args.dirs.copy()
    plot_multiple_average_curve(args.dirs, args.labels, args.title)

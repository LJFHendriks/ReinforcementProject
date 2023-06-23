import numpy as np
import bottleneck as bn
from matplotlib import pyplot as plt
from stable_baselines3.common import results_plotter
from glob import glob
import sys
import json
import os
from os import path

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    return bn.move_mean(values, window)[window - 1:]


def get_average(list):
    prev_shortest_len = 0
    result = [0 for _ in range(np.max([len(x) for x in list]))]
    while len(list) != 0:
        shortest_len = min([len(x) for x in list])
        shortest_item = np.argmin([len(x) for x in list])
        for item in list:
            result[prev_shortest_len:shortest_len] += item[prev_shortest_len:shortest_len]
            print(result[prev_shortest_len:shortest_len][0])
        result[prev_shortest_len:shortest_len] = np.divide(result[prev_shortest_len:shortest_len], len(list))
        list.pop(shortest_item)
        prev_shortest_len = shortest_len
    return result


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    monitor_files = glob(os.path.join(log_folder, "*"))
    x, y = [], []
    for folder in monitor_files:
        x_new, y_new = ts2xy(load_results(folder), "timesteps")
        y.append(y_new)
        x = x_new
    y_averaged = (get_average(y))
    y = moving_average(y_averaged, window=50)
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(log_dir + '/plot.png')


if __name__ == "__main__":
    log_dir = sys.argv[1]
    plot_results(log_dir)

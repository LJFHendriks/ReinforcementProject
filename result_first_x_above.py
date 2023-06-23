import warnings

import numpy as np
import os
import argparse

from stable_baselines3.common.monitor import load_results


def find_index(rewards, r, x):
    count = 0
    for i, rew in enumerate(rewards):
        if rew >= r:
            count += 1
        else:
            count = 0
        if count >= x:
            return i
    return -1


def first_x_above(dirs, rewards=None, x=None):
    rewards = [100., 200.] if rewards is None else rewards
    x = 10 if x is None else x
    dirs = [[os.path.join(dir, subdir) for subdir in os.listdir(dir)] for dir in dirs]
    times = [[list() for _ in rewards] for _ in dirs]
    std_times = np.full((len(dirs), len(rewards)), np.nan)
    mean_times = np.full((len(dirs), len(rewards)), np.nan)
    for i, log_dirs in enumerate(dirs):
        for dir in log_dirs:
            results = load_results(dir)
            for j, r in enumerate(rewards):
                index = find_index(results.r, r, x)
                times[i][j].append(sum(results.l[:index+1]))
        for j, _ in enumerate(rewards):
            if 0 not in times[i][j]:
                std_times[i][j] = np.std(times[i][j])
                mean_times[i][j] = np.mean(times[i][j])
    return mean_times, std_times


def to_latex(dirs, rewards, mean, std, divide=1e5):
    result = f'\\begin{{tabular}}{{l|{"r"*len(rewards)}}}\n'
    result += '\\toprule\n'
    result += 'reward'
    for r in rewards:
        result += f' & {r:.0f}'
    result += '\\\\ \n\\midrule \n'
    for i, name in enumerate(dirs):
        result += os.path.basename(name)
        for j, r in enumerate(rewards):

            result += format_value(mean[i][j], std[i][j], divide)
        result += ' \\\\ \n'
    result += '\\bottomrule \n'
    result += '\\end{tabular}'
    print(result)


def format_value(mean, std, divide):
    result = ' & '
    if np.isnan(mean):
        result += '-'
    else:
        result += f'{mean / divide:.1f}'
        if not np.isnan(std):
            result += f' \\pm {std / divide:.1f}'
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the first time step at which we have x runs above r reward')
    parser.add_argument('dirs', type=str, nargs='+')
    parser.add_argument('-r', '--reward', type=float, nargs='+')
    parser.add_argument('-x', type=int)

    args = parser.parse_args()
    args.dirs += [os.path.join('Combination', subdir) for subdir in os.listdir('Combination')]
    mean, std =  first_x_above(args.dirs, args.reward, args.x)
    print(mean)
    print(std)
    to_latex(args.dirs, args.reward, mean, std)

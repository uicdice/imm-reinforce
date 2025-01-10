import argparse
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--csv_dir', default='csv_cached')
args = parser.parse_args()

def trendFromCSVs(pathtemplate, uniques):
    trend_x = []
    trend_mean = []
    trend_low = []
    trend_high = []
    for unique in uniques:
        idx = round(unique, 1)
        trend_x.append(idx)
        readTable = pd.read_csv(pathtemplate.format(idx), header=None)
        trend = readTable.mean(axis=1)
        trend_mean.append(trend.mean())
        trend_low.append(trend.mean() - trend.quantile(0.1))
        trend_high.append(trend.quantile(0.9) - trend.mean())
    return trend_x, trend_mean, trend_low, trend_high

"""
Plot without lambda normalization.
"""

# Draw the 50 epochs trendline against varying lamdba_ratios
epochs50_x, epochs50_mean, epochs50_low, epochs50_high = trendFromCSVs(
    args.csv_dir + "/lambda_search/epochs=50,lambda_ratio={}.csv",
    np.linspace(0,0.9,10)
)
red = plt.plot(epochs50_x, epochs50_mean, '-o', c="darkred", label='50 Epochs')
plt.errorbar(epochs50_x, epochs50_mean, yerr=[epochs50_low, epochs50_high], capsize=7, fmt="o", c="darkred")

epochs200_x, epochs200_mean, epochs200_low, epochs200_high = trendFromCSVs(
    args.csv_dir + "/lambda_search/epochs=200,lambda_ratio={}.csv",
    np.linspace(0,0.9,10)
)
blue = plt.plot(epochs200_x, epochs200_mean, '-o', c="darkblue", label='200 Epochs')
plt.errorbar(epochs200_x, epochs200_mean, yerr=[epochs200_low, epochs200_high], capsize=7, fmt="o", c="darkblue")

epochs400_x, epochs400_mean, epochs400_low, epochs400_high = trendFromCSVs(
    args.csv_dir + "/lambda_search/epochs=400,lambda_ratio={}.csv",
    np.linspace(0,0.9,10)
)
green = plt.plot(epochs400_x, epochs400_mean, '-o', c="seagreen", label='400 Epochs')
plt.errorbar(epochs400_x, epochs400_mean, yerr=[epochs400_low, epochs400_high], capsize=7, fmt="o", c="seagreen")

plt.title("Average reward with varying lambda ratios.")
plt.legend(loc='lower right')
if shutil.which('latex'):
    plt.xlabel(r'$\frac{\lambda}{1+\lambda}$')
else:
    plt.xlabel("Lambda Ratio")
plt.ylabel("Average Reward")


plt.show()

import argparse
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--csv_dir', default='csv_cached')
args = parser.parse_args()

plt.rcParams.update({
    "text.usetex": True if shutil.which('latex') else False,
    "font.size": 14
})

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

for ds in [50, 200, 400]:
    baseline_x, baseline_mean, baseline_low, baseline_high = trendFromCSVs(
        args.csv_dir + "/lambda_search/epochs=" + str(ds) + ",lambda_ratio=0.0.csv",
        np.linspace(0,0.9,10)
    )

    imm_x, imm_mean, imm_low, imm_high = trendFromCSVs(
        args.csv_dir + "/lambda_search/epochs=" + str(ds) + ",lambda_ratio={}.csv",
        np.linspace(0,0.9,10)
    )
    plt.figure()

    red = plt.plot(baseline_x, baseline_mean, '-o', c="darkred", label=f'no IMM')
    plt.errorbar(baseline_x, baseline_mean, yerr=[baseline_low, baseline_high], capsize=7, fmt="o", c="darkred")

    green = plt.plot(imm_x, imm_mean, '-o', c="seagreen", label=f'IMM against maximal utility action')
    plt.errorbar(imm_x, imm_mean, yerr=[imm_low, imm_high], capsize=7, fmt="o", c="seagreen")

    plt.title(f"Average reward after training for {ds} epochs")
    plt.legend(loc='lower right')
    plt.xlabel(r'$\frac{\lambda}{1+\lambda}$')
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig(f"ds={ds}.pdf", bbox_inches='tight')


# plt.show()

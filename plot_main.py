import argparse
import glob

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
        single_file_path = pathtemplate.format(idx)
        single_file_path = glob.glob(single_file_path)[0] # needed to resolve wildcard used for lambda
        readTable = pd.read_csv(single_file_path, header=None)
        trend = readTable.mean(axis=1) # get average reward from multiple rollouts
        trend_mean.append(trend.mean())
        trend_low.append(trend.mean() - trend.quantile(0.1))
        trend_high.append(trend.quantile(0.9) - trend.mean())
    return trend_x, trend_mean, trend_low, trend_high

imm_x, imm_mean, imm_low, imm_high = trendFromCSVs(
    # "./outputs_epochs/epochs={},imm=True.csv",
    args.csv_dir + "/imm_pomdp_max_utility_action/epochs={},lambda_ratio=*.csv",
    [10, 20, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
)
green = plt.plot(imm_x, imm_mean, '-o', c="seagreen", label='IMM using maximal utility action')
plt.errorbar(imm_x, imm_mean, yerr=[imm_low, imm_high], capsize=4, fmt="o", c="seagreen")

imm_pomdp_t05_x, imm_pomdp_t05_mean, imm_pomdp_t05_low, imm_pomdp_t05_high = trendFromCSVs(
    # "./outputs_pomdp_softmax=0.5/epochs={},lambda_ratio=None.csv",
    args.csv_dir + "/imm_pomdp_softmax_temp_0.5/epochs={},lambda_ratio=*.csv",
    [10, 20, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
)
blue = plt.plot(imm_pomdp_t05_x, imm_pomdp_t05_mean, '-x', c="darkblue", label='IMM using softmaxed POMDP policy (temperature=0.5)')
plt.errorbar(imm_pomdp_t05_x, imm_pomdp_t05_mean, yerr=[imm_pomdp_t05_low, imm_pomdp_t05_high], capsize=4, fmt="x", c="darkblue")



imm_pomdp_t10_x, imm_pomdp_t10_mean, imm_pomdp_t10_low, imm_pomdp_t10_high = trendFromCSVs(
    # "./outputs_pomdp_softmax=1.0/epochs={},lambda_ratio=None.csv",
    args.csv_dir + "/imm_pomdp_softmax_temp_1.0/epochs={},lambda_ratio=*.csv",
    [10, 20, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
)
blue = plt.plot(imm_pomdp_t10_x, imm_pomdp_t10_mean, '--o', c="#e28e1f", label='IMM using softmaxed POMDP policy (temperature=1.0)')
plt.errorbar(imm_pomdp_t10_x, imm_pomdp_t10_mean, yerr=[imm_pomdp_t10_low, imm_pomdp_t10_high], capsize=4, fmt="o", c="#e28e1f")

# Draw the 50 epochs trendline against varying lamdba_ratios
baseline_x, baseline_mean, baseline_low, baseline_high = trendFromCSVs(
    args.csv_dir + "/none/epochs={},lambda_ratio=0.0.csv",
    [10, 20, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500]
)
red = plt.plot(baseline_x, baseline_mean, '--x', c="darkred", label='No IMM')
plt.errorbar(baseline_x, baseline_mean, yerr=[baseline_low, baseline_high], capsize=4, fmt="x", c="darkred")

# plt.title("Effect of quality of restricted model on IMM performance")
plt.xlabel("Number of epochs")
plt.ylabel("Average Reward")

plt.legend(loc='lower right')
plt.show()
# plt.savefig("mdp_plots.pdf", bbox_inches='tight')

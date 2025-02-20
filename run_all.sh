#!/bin/bash

# We train TOTAL_RUNS times but evaluate the policy 10 times
# for each training.
# Number of Monte Carlo runs per curve is therefore 10x this
# number
TOTAL_RUNS=30

# Obtain the number of logical CPUs on your computer by running
# getconf _NPROCESSORS_ONLN

# PARALLEL_JOBS may be slightly higher than the number of online
# processors (i.e. it's okay to overschedule)
# but it should EXACTLY divide TOTAL_RUNS
# e.g. if we have 12 CPUs, we will set it to 15
# and if we have 32 CPUs, we will set it to 30
PARALLEL_JOBS=30

echo "Scheduling baseline runs"
source schedulers/run_baseline.sh

echo "Scheduling IMM runs with Bayes Optimal target model"
source schedulers/run_imm_model=bayes_optimal.sh

echo "Scheduling IMM runs with Medium Quality target model"
source schedulers/run_imm_model=medium_quality.sh

echo "Scheduling IMM runs with Low Quality target model"
source schedulers/run_imm_model=low_quality.sh

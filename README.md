## IMM in training MDPs

This is the last of the three repositories accompanying the paper *Induced Model Matching: Restricted Models Help Train Full-Featured Models (NeurIPS 2024)*

```bibtex
@inproceedings{muneeb2024induced,
    title     = {Induced Model Matching: Restricted Models Help Train Full-Featured Models},
    author    = {Usama Muneeb and Mesrob I Ohannessian},
    booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year      = {2024},
    url       = {https://openreview.net/forum?id=iW0wXE0VyR}
}
```

This repository serves a simple example where POMDP policies can be used to improve the training of MDP policies via Induced Model Matching (IMM).

Other repositories: [IMM in Logistic Regression](https://github.com/uicdice/imm-logistic-regression) | [IMM in Language Modeling](https://github.com/uicdice/imm-language-modeling)

We first define the MDP 5-tuple $$\langle \mathcal{S}, \mathcal{A}, T, R, \gamma \rangle$$. We consider a simple 11x11 grid with a mountain shaped reward landscape, having a peak in the center. In each of the 121 possible states (i.e. coordinates), the agent may take act to move in one of four possible directions (the grid is a toroid and wraps around). The agent has to learn to navigate a simple grid like environment where the reward distribution is shaped like a mountain (with a peak in the center). We assume a continuing task and the agent's goal is to not just reach the peak but also stay there. We employ a non-zero discount factor $\gamma$.

The POMDP requires definition of the observation space and function ($$\mathcal{O}$$ and $$O$$) in addition to the MDP 5-tuple. For this proof of concept, we assume a degenerate observation model, i.e., if an agent is at coordinate $x=7,y=3$, we can only observe $x$ coordinate to be 7 (and not any other value). The $y$ coordinate is not observable and we maintain a uniform belief over the 11 possible values.

### Generating an accurate POMDP solution

We use the POMDPs.jl package in combination with the Fast Informed Bound (FIB.jl) POMDP solver to accomplish this. Please first install Julia and then the required packages by running:

```bash
julia install_packages.jl
```

To obtain the accurate restricted model, we model the POMDP and solve it via FIB to obtain the $$\alpha$$-vectors. To do so, please run:

```bash
julia simplegrid_pomdp.jl
```

### IMM aided training

We recommend running this in a CPU only PyTorch environment (if GPU version is used, CUDA will need to be initialized repeatedly, eventually slowing things down). The following additional packages will be required:

```bash
pip install gymnasium h5py
```

We derive from the reward-to-go formulation (`2_rtg_pg.py` file) from OpenAI Spinning Up documentation and add the IMM component. The file is MIT licensed and a copy has been included in the `openai` folder for reference.

#### Determining optimal $$\lambda$$ for a dataset size

Just as we did for the logistic regression experiments, we determine an optimal $$\lambda$$ schedule via cross validation. These schedules have already been incorporated into `main.py` and you may skip this, but is being included for completeness. During cross validation, we directly control not $$\lambda$$ but rather the ratio $$\frac{\lambda}{1+\lambda}$$ which is the contribution of the IMM coefficient as a fraction of both IMM and main objective coefficients.

This should be done separately for each configuration on the main plot (we have three configurations, one using argmax or maximal utility based POMDP policy and two using softmax based policies). For example, the following snippet will generate the results needed to tune the $$\lambda$$ schedule for the configuration performing IMM done against the target POMDP policy that uses maximal utility action (which is the default configuration).

```bash
export dss=(50 200 400)
export lambda_ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for ds in ${dss[@]}; do
    for lambda_ratio in ${lambda_ratios[@]}; do
        for seed in {1..30}; do
            python main.py --epochs $ds --batch_size 50 --gamma 0.9 --seed $seed --lambda_ratio $lambda_ratio --output_dir csv/lambda_search
        done
    done
done
```

The results can be visualized by running `python plot_lambda_tuning.py`. By default, it will read cached CSVs from the `csv_cached` directory. Another source can be specified through an additional argument, i.e. `python plot_lambda_tuning.py --csv_dir csv`.

We experimentally determined the same rule to also apply in the case of softmaxed POMDP policies (i.e. with additional `--pomdp_temp 0.5` or `--pomdp_temp 1.0` argument). The rule has been integrated into `main.py` and will be applied if we specify ` --lambda_ratio -1` to automatically set the $$\lambda$$ parameter.

#### Performance with IMM using different restricted targets

All three curves shown on our main plot can be generated using this BASH snippet. Precisely, we show a curve with no IMM component, and three curves with IMM aided training against models of different qualities.

For each, we will use the $$\lambda$$ rule determined in the previous step, which precisely is $$\frac{\lambda}{1+\lambda} = 0.0002 n + 0.1891$$ where $$n$$ is the dataset size (analogous to number of epochs in this experiment).

```bash
export dss=(10 20 50 75 100 125 150 175 200 250 300 350 400 450 500)
for ds in ${dss[@]}; do
    for seed in {1..30}; do
        python main.py --epochs $ds --batch_size 50 --gamma 0.9 --seed $seed --lambda_ratio 0.0 --output_dir csv/none
        python main.py --epochs $ds --batch_size 50 --gamma 0.9 --seed $seed --lambda_ratio -1 --pomdp_temp 1.0 --output_dir csv/imm_pomdp_softmax_temp_1.0
        python main.py --epochs $ds --batch_size 50 --gamma 0.9 --seed $seed --lambda_ratio -1 --pomdp_temp 0.5 --output_dir csv/imm_pomdp_softmax_temp_0.5
        python main.py --epochs $ds --batch_size 50 --gamma 0.9 --seed $seed --lambda_ratio -1 --output_dir csv/imm_pomdp_max_utility_action
    done
done
```

The results can be visualized by running `python plot_main.py`. Again, by default it will read cached CSVs from the `csv_cached` directory. Another source can be specified through an additional argument, i.e. `python plot_lambda_tuning.py --csv_dir csv`.

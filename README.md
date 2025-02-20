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

> [!NOTE]
> **The MDP**  We first define the MDP 5-tuple $$\langle \mathcal{S}, \mathcal{A}, T, R, \gamma \rangle$$. We consider a simple 11x11 grid with a mountain shaped reward landscape, having a peak in the center. In each of the 121 possible states (i.e. coordinates), the agent may take act to move in one of four possible directions (the grid is a toroid and wraps around). The agent has to learn to navigate a simple grid like environment where the reward distribution is shaped like a mountain (with a peak in the center). We assume a continuing task and the agent's goal is to not just reach the peak but also stay there. We employ a non-zero discount factor $\gamma$.
> 
> **The POMDP** The POMDP requires definition of the observation space and function ($$\mathcal{O}$$ and $$O$$) in addition to the MDP 5-tuple. For this proof of concept, we assume a degenerate observation model, i.e., if an agent is at coordinate $x=7,y=3$, we can only observe $x$ coordinate to be 7 (and not any other value). The $y$ coordinate is not observable and we maintain a uniform belief over the 11 possible values.

### Quick Start

The `main.py` (if run with all the default options), will train a policy without any restricted model information for 200 epochs. It will then evaluate the policy using 10 rollouts. In order to replicate each of the reported curves in the paper, there are options provided in this file that can be set accordingly. We derive `main.py` from the *reward-to-go* formulation (`2_rtg_pg.py` file) from OpenAI Spinning Up documentation and add the IMM component. The file is MIT licensed and a copy has been included in the `openai` folder for reference.

Since the plots require 30 Monte Carlo training runs for each configuration, and running them sequentially is time consuming, we have provided a `run_all.sh` BASH script that will parallelize these 30 runs using multiprocessing. Note that **$$\alpha$$-vectors need to be generated before calling `run_all.sh`** (details in next section).

Alternatively, if you simply want to generate plots from cached files, you can skip directly to the [plotting](#plotting-generated-csvs) section below (cached CSV files for 300 runs have also been provided in this repository).

> [!IMPORTANT]
> **Secondary Objective Coefficient** One of `--lambda_ratio` or `--lambda_param` parameters can be used to set the coefficient for the secondary objective (i.e. IMM or noising). While `--lambda_param` will set $$\lambda$$ directly, `--lambda_ratio` will set $$\frac{\lambda}{1+\lambda}$$ (from which `train_and_test.py` will determine $$\lambda$$). Controlling $$\frac{\lambda}{1+\lambda}$$ is more intuitive, since it is the contribution of the secondary objective's coefficient as a fraction of both main as well as secondary objective coefficients.
> 
> **Overall Objective** The overall objective function minimized is *policy gradient objective* + $$\lambda$$ *IMM using POMDP policy objective*. Unlike [logistic regression](https://github.com/uicdice/imm-logistic-regression) experiments, we **do not** normalize our learning rate by $$1+\lambda$$ in the code, however, we would like to point out that we use Adam Optimizer in these experiments (unlike SGD in logistic regression experiments).

### Generating an accurate POMDP solution

We have also provided a cached version of $$\alpha$$-vectors in this repository. You may choose to skip this step and use the cached version, however, **make sure to rename them so the training script can find them**:

```bash
mv alphas_cached.h5 alphas.h5
```

If you want to generate $$\alpha$$-vectors by solving the POMDP, we will use the POMDPs.jl package in combination with the Fast Informed Bound (FIB.jl) POMDP solver to accomplish this. Please first install Julia and then the required packages by running:

```bash
julia install_packages.jl
```

To obtain the accurate restricted model, we model the POMDP and solve it via FIB to obtain the $$\alpha$$-vectors. To do so, please run:

```bash
julia simplegrid_pomdp.jl
```

### IMM aided training

We recommend running this in a CPU only PyTorch environment (if GPU version is used, CUDA will need to be initialized repeatedly, eventually slowing things down). To this end, the official PyTorch Docker container (`pytorch/pytorch`) is recommended. Also, the following additional packages will be required:

```bash
pip install gymnasium h5py
```

To maximally utilize multiprocessing, you are encouraged to edit `PARALLEL_JOBS` inside `run_all.sh`, as per the provided instructions before calling it as:

```bash
./run_all.sh
```

### Plotting Generated CSVs

The results can be visualized by running `python plot_main.py`. 

By default, the cached CSVs in `csv_cached` directory will be used. This can be overridden by adding an extra `--csv_dir csv` flag.

### Obtaining schedules for $$\lambda$$

For completeness, we will also demonstrate how we obtain the plots in the paper that we use to create the hardcoded rules that give us $$\lambda$$ from the dataset size. Directly, the relationship with the dataset size is not that of $$\lambda$$ but that of $$\frac{\lambda}{1+\lambda}$$.

```bash
./schedulers/tune_lambda.sh
```

To visualize the generated CSVs

```bash
python plot_lambda_search.py
```

Again, by default, the cached CSVs in `csv_cached/lambda_search` directory will be used. This can be overridden by adding an extra `--csv_dir csv/lambda_search` flag.

We experimentally determined the same rule to also apply in the case of softmaxed POMDP policies (i.e. with additional `--pomdp_temp 0.5` or `--pomdp_temp 1.0` argument). The rule has been integrated into `main.py` and will be applied if we specify ` --lambda_ratio -1` to automatically set the $$\lambda$$ parameter.



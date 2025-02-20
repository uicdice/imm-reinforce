r"""
This is the third of the three repositories accompanying the paper

    Induced Model Matching: Restricted Models Help Train Full-Featured Models (NeurIPS 2024)

This repository demonstrates Induced Model Matching to incorporate a POMDP policy's knowledge into
the training of a "full featured" MDP policy.

Other implementations are contained in the following repositories:

    IMM in Logistic Regression: https://github.com/uicdice/imm-logistic-regression
    IMM in Language Modeling: https://github.com/uicdice/imm-language-modeling

This file has been adapted from the vanilla policy gradient code `2_rtg_pg.py` provided in
OpenAI Spinning Up documentation. A copy of the original has been included in the `openai` folder.

Copyright 2023-2024 Usama Muneeb and Mesrob Ohannessian
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.envs.registration import register
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from simplegrid import action_to_symbol

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', '--env', type=str, default='SimpleGridH11W11')
parser.add_argument('--render', action='store_true')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_rollouts', type=int, default=10)
parser.add_argument('--eval_steps', type=int, default=22)

parser.add_argument('--rewardshape', choices=['mountain', 'delta'], default='mountain')
parser.add_argument('--gridwraparound', action='store_false')
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--transfer_type', choices=['imm', 'interpolate'], default='imm')
parser.add_argument('--lambda_ratio', type=float, default=0.0)
parser.add_argument('--lambda_param', type=float)
parser.add_argument('--seed', type=int, required=False)
parser.add_argument('--pomdp_temp', type=float, default=-1)
parser.add_argument('--output_dir', type=str, default='outputs')
args = parser.parse_args()

r"""
Determine IMM coefficient.

Since we minimize

    1 * Main Objective + \lambda * IMM Objective

the direct relationship with the dataset size is that of lambda / (1+lambda)
which we term the "ratio" between the coefficients of the two components.

We therefore allow the user to specify the --lambda_ratio parameter for searching the right
lambda rule (parameter tuning experiments).

Additionally, we allow the directly specifying --lambda_param parameter for performing ablation
experiments against dataset size.

We have determined the right lambda schedule to be

    ratio = -0.0002 * dataset_size + 0.1891

for all IMM curves on our main plot. This rule is hardcoded into this file and if the user
would like to use this rule, --lambda_ratio -1 can be specified.

The output file will be named according to the argument used (lambda_ratio or lambda_param).

NOTE: eventually lambda_param is used in the code.
NOTE: dataset size in these experiments is equivalent to number of epochs, for reasons described
in the paper.
NOTE: looking at the main objective above, if we use SGD, we would need to normalize learning rate
with (1+lambda). We however use Adam Optimizer and this isn't needed.
NOTE: --lambda_ratio 0 or --lambda_param 0 must be specified to obtain baseline without IMM or
interpolation
NOTE: --lambda_ratio -1 is supported, --lambda_param -1 is not!
"""

if args.lambda_ratio is not None:
    if args.lambda_ratio < 0:
        # lambda schedule determined using parameter tuning
        ratio = round(-0.0002 * args.epochs + 0.1891, 1)
        lambda_param = ratio/(1 - ratio)
        lambda_in_filename = f"lambda_ratio={ratio:.1f}"
    else:
        """
        We do the inverse of
            ratio = lambda / (1 + lambda)

        to get the lambda from the ratio:

            ratio*(1+lambda) = lambda
            ratio + ratio*lambda = lambda
            ratio = lambda - ratio*lambda
            lambda (1 - ratio) = ratio
            lambda = ratio/(1 - ratio)
        """
        lambda_param = args.lambda_ratio/(1 - args.lambda_ratio)
        lambda_in_filename = f"lambda_ratio={args.lambda_ratio:.1f}"

elif args.lambda_param is not None:
    lambda_param       = args.lambda_param
    lambda_in_filename = f"lambda_param={args.lambda_param:.1f}"


if args.seed is not None:
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

def softmax_with_temperature(logits, temperature=1.0):
    logits /= temperature
    softmax_result = F.softmax(logits, dim=-1)
    return softmax_result

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

GRID_Y_LENGTH = 11 # y-coordinate will not be known to POMDP (we will maintain uniform belief here)
GRID_X_LENGTH = 11 # x-coordinate will be known to POMDP

register(
    id = "SimpleGridH11W11",
    entry_point = "simplegrid:SimpleGrid",
    kwargs = {
        "height": GRID_Y_LENGTH,
        "width": GRID_X_LENGTH,
        "shape": args.rewardshape,
        "wraparound": args.gridwraparound
    }
)

env_supports_partial_obs = True

if lambda_param >= 0:
    # the restricted model (POMDP plan) is needed for both IMM and interpolation
    import h5py
    with h5py.File("alphas.h5", 'r') as hdf5_file:
        alphas = hdf5_file['alphas'][:]

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        # for the last layer, use `nn.Identity`
        # for all others, use `nn.Tanh`
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class mlpModule(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Identity, params=None):
        super(mlpModule, self).__init__()

        self.layers = []
        for j in range(len(sizes)-1):
            # for the last layer, use `nn.Identity`
            # for all others, use `nn.Tanh`
            act = activation if j < len(sizes)-2 else output_activation
            self.layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

        if params:
            del self.layers[0].weight
            del self.layers[0].bias
            del self.layers[2].weight
            del self.layers[2].bias

            # reuse these parameters for this copy of the forward graph
            self.layers[0].weight = params[0][0]
            self.layers[0].bias = params[0][1]

            self.layers[2].weight = params[1][0]
            self.layers[2].bias = params[1][1]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

    def get_params(self):
        params = [(x.weight,x.bias) for x in self.layers if hasattr(x,'weight')]
        return params

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def discounted_reward(rews, gamma=0.9):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + gamma * (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def prw(rews):
    print(["%.2f" % x for x in rews])

def train(env_name='SimpleGridH11W11', hidden_sizes=[32], lr=1e-2, 
          epochs=200, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)

    # we represent state as a `spaces.Tuple` object (tuple of y and x coordinate)
    # it's converted to one hot just being fed to policy network
    obs_dim = env.observation_space[0].n * env.observation_space[1].n
    n_acts = env.action_space.n
    print(f"obs_dim: {obs_dim}")
    print(f"n_acts: {n_acts}")

    # make core of policy network
    print("Creating forward graph for policy network")
    logits_net = mlpModule(sizes=[obs_dim]+hidden_sizes+[n_acts])

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    softmax = nn.Softmax(1)

    num_random_samples = GRID_Y_LENGTH
    logits_net_samples = []
    print(f"Unobserved state variable has {GRID_Y_LENGTH} possible values")
    print(f"Assuming uniform belief over {GRID_Y_LENGTH} possible values")
    print(f"Creating {GRID_Y_LENGTH} additional forward graphs for computing induced POMDP")
    for _ in range(num_random_samples):
        sample = mlpModule(sizes=[obs_dim]+hidden_sizes+[n_acts], params=logits_net.get_params())
        logits_net_samples.append(sample)


    def visualize_policy():
        batch_obs = np.eye(GRID_Y_LENGTH * GRID_X_LENGTH) # gives one hot of all possible states

        logits = logits_net(torch.Tensor(batch_obs))
        categorical = Categorical(logits=logits)
        probs = categorical.probs
        actions = np.argmax(probs.detach().numpy(), axis=1)
        return np.reshape(actions, [GRID_Y_LENGTH, GRID_X_LENGTH])

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs, pomdp_utilities=None, lambda_interpolation=0.0):
        """Takes an observation (and optionally, also POMDP utilities) and returns an action.
        The POMDP utilities are used if interpolating policy network with restricted model.
        """
        # if False: # only use restricted model
        #     return np.argmax(pomdp_utilities,axis=1) # just return the POMDP action

        obs = obs.flatten()
        logits = logits_net(obs)

        # For interpolation, we will also provide the restricted model to interpolate with.
        if pomdp_utilities is not None:
            mdp_policy = softmax_with_temperature(logits, 1)
            combined_policy = (1-lambda_interpolation)*mdp_policy

            if args.pomdp_temp > 0: # use softmaxed POMDP
                pomdp_policy = softmax_with_temperature(logits, args.pomdp_temp)
                combined_policy += lambda_interpolation * pomdp_policy
            else: # use argmaxed POMDP
                pomdp_action = np.argmax(utilities,axis=1)
                combined_policy[pomdp_action] += lambda_interpolation

            return Categorical(
                probs=combined_policy # already softmaxed, providing as `probs` instead of `logits`
            ).sample().item()

        else:
            categorical = Categorical(logits=logits)
            return categorical.sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        """
        NOTE: instead of loss, a better name may be (negative) performance metric.
        """
        obs = obs.reshape([obs.shape[0],-1])
        logits = logits_net(obs)
        categorical = Categorical(logits=logits)

        logp = categorical.log_prob(act)
        return -(logp * weights).mean()





    """
    The following additional routines have been added for the IMM component.
    """
    def convert_obs_to_onehot(obs):
        grid = np.zeros((GRID_Y_LENGTH, GRID_X_LENGTH))
        grid[obs[0], obs[1]] = 1.0
        return grid

    def utilities_from_partial_obs(partial_obs):
        """This routine gives us the *accurate target model*.

        The x-component is the partial observation (and the equivalent of "restricted feature set")
        We compute a (uniform) belief vector over all (x, y) pairs corresponding to this x-coord.

        We will then compute the inner product of this belief with alpha-vectors to get the utility
        for each action (there is one alpha-vector per action).
        POMDP policy should be a softmax over the utilities of each action: UP, DOWN, LEFT, RIGHT.
        """

        # compute belief function using x-coordinate
        belief = np.zeros((len(partial_obs), GRID_Y_LENGTH, GRID_X_LENGTH))
        indices = np.vstack(list(enumerate(partial_obs)))
        belief[indices[:,0],:,indices[:,1]] = 1 / belief.shape[1]

        # take inner product of belief with alpha-vectors
        utilities = np.matmul(belief.reshape([-1, GRID_Y_LENGTH * GRID_X_LENGTH]), alphas)
        return utilities

    def gen_multiset(batch_obs, batch_partial_obs):
        """Generates the extended multiset (both x and y-coordinates) for a fixed x-coordinate.
        This is equivalent to "random sampling" extended features while holding the
        "restricted feature set" fixed.

        The x-component is the partial observation (and the equivalent of "restricted feature set")

        We will need the multiset to get the induced POMDP policy.
        """

        num_random_samples = GRID_Y_LENGTH # since we maintain the belief over y component
        sampled_batch_dims = len(batch_obs)
        sample_dims = batch_obs[0].shape # 3
        sample_dtype = batch_obs[0].dtype

        # The size of `multiset` is
        # BATCH_SIZE, NUM_SAMPLES (GRID_Y_LENGTH), GRID_Y_LENGTH, PARTIAL_OBS (X_COORD)
        # NOTE: GRID_Y_LENGTH happens twice because y-coordinate is used twice.
        # Used first to index the sample, and then again to "one hot" the actual y-coord

        multiset = np.zeros(
            [sampled_batch_dims, num_random_samples] + list(sample_dims),
            sample_dtype
        )

        idx = np.vstack(list(enumerate(batch_partial_obs)))
        for j in range(num_random_samples):
            multiset[idx[:,0], j, j, idx[:,1]] = 1

        return multiset

    def get_imm_loss(obs, partial_obs, act, weights):
        """Compute MDP's induced POMDP policy and match against provided POMDP solution.
        """

        # Ingredient 1: Accurate POMDP policy
        # POMDP policy should only take subset of the state as input

        utilities = utilities_from_partial_obs(partial_obs)

        # Ingredient 2: Induced POMDP policy from MDP policy
        # Need to compute multiset (required for induced POMDP policy)

        multiset = gen_multiset(obs, partial_obs)

        num_random_samples = GRID_Y_LENGTH
        q_y = []
        for j in range(num_random_samples):
            sample = torch.as_tensor(multiset[:,j,:,:], dtype=torch.float32)
            sample_logits = logits_net_samples[j](sample.reshape([sample.shape[0],-1]))

            q_y.append(Categorical(logits=sample_logits).probs)

        hatQ = sum(q_y) / num_random_samples

        if args.pomdp_temp > 0:
            hatP = softmax_with_temperature(torch.Tensor(utilities), args.pomdp_temp)
        else:
            hatP = torch.Tensor(np.identity(alphas.shape[1])[
                np.argmax(utilities,axis=1)
            ])
        imm_loss = kl_loss(torch.log(hatQ), hatP)
        return imm_loss

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_partial_obs = []
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()[0]       # first obs comes from starting distribution
        if env_supports_partial_obs:
            partial_obs = obs[1] # only `x` is visible

        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # pdb.set_trace()
        # collect experience by acting in the environment with current policy
        for i in range(batch_size):

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render() # NOTE: `render` is not currently implemented for simple grid

            # save obs
            batch_obs.append(convert_obs_to_onehot(obs))
            if env_supports_partial_obs:
                batch_partial_obs.append(partial_obs)

            # act in the environment
            act = get_action(
                torch.as_tensor(convert_obs_to_onehot(obs),
                dtype=torch.float32)
            )
            obs, rew, done, _, _ = env.step(act)

            # for a non-episodic (i.e. continuing `env`), set `done` to `True` when `batch_size`
            # observations achieved (otherwise it will never be "done")
            if i==batch_size-1:
                done = True

            # NOTE: if this is the last one for this episode (and `done` hits below),
            # then this `obs` will be wasted.
            # This is because `batch_obs.append(...)`` is called at the beginning of the loop and
            # the `obs` obtained later from `step` and `reset` are appended in the next iteration.

            if env_supports_partial_obs:
                partial_obs = obs[1] # only `x` is visible

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is (discounted) reward-to-go from t
                batch_weights += list(discounted_reward(ep_rews, args.gamma))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []
                if env_supports_partial_obs:
                    partial_obs = obs[1] # only `x` is visible

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        if args.transfer_type=='imm' and lambda_param > 0:
            imm_loss = lambda_param * \
                get_imm_loss(batch_obs, batch_partial_obs, batch_acts, batch_weights)
            imm_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        if i % 10 == 0:
            for row in visualize_policy():
                newrow = list(map(lambda x : action_to_symbol[x], row))
                print(" ".join([u'%s' % val for val in newrow]))

    if args.transfer_type=='interpolate':
        # For interpolation, we need multiple evaluation runs for the same trained model.
        eval_configs = np.linspace(0,0.9,10)# NOTE: this is lambda ratio, not lambda!
    elif args.transfer_type=='imm':
        # For IMM, all we want is a single evaluation; the (single) 0 is just a placeholder.
        eval_configs = [0]

    for lambdaratio in eval_configs:
        # In interpolation we have multiple lambda (for evaluation) per same trained model
        # this will give us the same level of reproducibility as IMM (where we had only one lamdba)
        # this ensures sequence of lambda does not affect reproducibility
        if args.transfer_type=='interpolate':
            np.random.seed(args.seed)
            torch.random.manual_seed(args.seed)

        # Evaluation loop (compute cumulative reward)
        # For interpolation, filename is independent of provided --lambda_ratio or --lambda_param
        # and the `lambda_in_filename` set earlier is not used.
        # This is because it's trivial to run for all lambda ratios (i.e. eval_configs) for the
        # same training
        if args.transfer_type=='imm':
            myfile = open(
                os.path.join(args.output_dir, f"epochs={args.epochs},{lambda_in_filename}.csv"),
                "a"
            )
        elif args.transfer_type=='interpolate':
            myfile = open(
                os.path.join(args.output_dir, f"epochs={args.epochs},lambda_ratio={lambdaratio:.1f}.csv"),
                "a"
            )
        reward_all_rollouts = 0
        for i in range(args.num_rollouts):
            total_reward = 0
            obs = env.reset()[0]       # first obs comes from starting distribution
            for j in range(args.eval_steps):
                if args.transfer_type=='interpolate':
                    partial_obs = obs[1]
                    utilities = utilities_from_partial_obs([partial_obs])
                    act = get_action(
                        torch.as_tensor(convert_obs_to_onehot(obs), dtype=torch.float32),
                        # `utilities_from_partial_obs`, when used during training,
                        # expects a batch and returns a batch.
                        # When using it in evaluation mode, the batch size is 1.
                        torch.as_tensor(utilities[0], dtype=torch.float32),
                        lambdaratio
                    )
                elif args.transfer_type=='imm':
                    act = get_action(
                        torch.as_tensor(convert_obs_to_onehot(obs), dtype=torch.float32)
                    )
                obs, rew, done, _, _ = env.step(act)
                total_reward += rew
                if done:
                    break

            print(f"Rollout {i}: Reward: {total_reward}, Ran for {args.eval_steps} steps")
            reward_all_rollouts += total_reward

            myfile.write(f"{total_reward:.3f},")

        myfile.write("\n")

train(env_name=args.env_name, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)

from enum import IntEnum
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType


class Actions(IntEnum):
    # movement
    # these numbers determine what each index of the policy PMF looks like
    # these should correspond to what we have in the POMDP implementation
    # the order followed there is UP DOWN LEFT RIGHT
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

action_to_symbol = {
    Actions.UP: '\u25B2',
    Actions.DOWN: '\u25BC',
    Actions.LEFT: '\u25C4',
    Actions.RIGHT: '\u25BA',
}

# We use the Gaussian distribution functions to mimic a "mountain like" reward landscape
# For technical reasons, the word "Gaussian" or "Normal" should not be used to describe
# the reward landscape, instead "mountain like" should be used.
def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N


class SimpleGrid(gym.Env):
    def __init__(self, height, width, shape='mountain', wraparound=False):
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.width = width
        self.height = height

        self.agent_x = None
        self.agent_y = None

        # self.terminal_x = self.width // 2
        # self.terminal_y = self.height // 2

        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height), spaces.Discrete(self.width)
        ))

        # needed only if we use the `truncated` flag
        # we do not use `truncated` in our code, instead we use the `terminated` flag
        self.max_steps = 256

        self.wraparound = wraparound

        if shape=='mountain':
            N = self.width
            X = np.linspace(-2, 2, self.width)
            Y = np.linspace(-2, 2, self.height)
            X, Y = np.meshgrid(X, Y)

            # Mean vector and covariance matrix
            mu = np.array([0., 0.])
            Sigma = np.array([[ 0.1 , 0], [0,  0.1]])

            # Pack X and Y into a single 3-dimensional array
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y

            self.reward = multivariate_gaussian(pos, mu, Sigma)

        elif shape=='delta':
            self.reward = np.zeros((self.height, self.width))
            self.reward[self.height // 2, self.width // 2] = 1.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.agent_x = np.random.randint(0, self.width)
        self.agent_y = np.random.randint(0, self.height)

        self.step_count = 0

        agent_loc = (self.agent_y, self.agent_x)
        return agent_loc, {}



    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if self.wraparound:
            if action == self.actions.RIGHT:
                self.agent_x = (self.agent_x + 1) % self.width
            elif action == self.actions.DOWN:
                self.agent_y = (self.agent_y + 1) % self.height
            elif action == self.actions.LEFT:
                self.agent_x = (self.agent_x - 1) % self.width
            elif action == self.actions.UP:
                self.agent_y = (self.agent_y - 1) % self.height
        else:
            if action == self.actions.RIGHT:
                self.agent_x = min(self.agent_x + 1, self.width-1)
            elif action == self.actions.DOWN:
                self.agent_y = min(self.agent_y + 1, self.height-1)
            elif action == self.actions.LEFT:
                self.agent_x = max(self.agent_x - 1, 0)
            elif action == self.actions.UP:
                self.agent_y = max(self.agent_y - 1, 0)

        self.step_count += 1

        # never terminate, the agent must stay near the peak
        # NOTE: this makes it a continuing task and in the code, we artificially terminate
        # by setting the `done` to True after reaching `batch_size=5000` observations
        # `done` is old name for `terminated` in the specification for `env.step()`
        if False: # self.agent_x == self.terminal_x and self.agent_y == self.terminal_y:
            terminated = True
        else:
            terminated = False

        agent_loc = (self.agent_y, self.agent_x)

        reward = self.reward[self.agent_y, self.agent_x]

        # NOTE: `truncated` status is not used in `main.py`
        if self.step_count >= self.max_steps:
            truncated = True
        else:
            truncated = False

        return agent_loc, reward, terminated, truncated, {}

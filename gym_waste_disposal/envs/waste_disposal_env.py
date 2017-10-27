"""Waste Disposal Environment"""
import io
import sys

import gym
import numpy as np
import scipy.stats
from gym import spaces
from gym.utils import seeding


class WasteDisposalEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.storage_cost = 0.01

        self.market_period = 20
        self.market_price_mean = 2
        self.market_price_amplitude = 1
        self.market_noise_scale = 0.1

        self.inspection_period = 7
        self.inspection_rate_mean = 0.05
        self.inspection_rate_amplitude = 0.05

        self.pollution_decay = 0.8
        self.pollution_threshold = 0.5
        self.pollution_penalty = 5000

        self.action_space = gym.spaces.Discrete(3)
        # yapf: disable
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.inf, shape=(1,)),
            spaces.Box(low=(self.market_price_mean
                            - self.market_price_amplitude
                            - 2 * self.market_noise_scale),
                       high=(self.market_price_mean
                             + self.market_price_amplitude
                             + 2 * self.market_noise_scale),
                       shape=(1,)),
            spaces.Discrete(2),
            spaces.Discrete(2)
        ))
        # yapf: enable
        self.reward_range = (-np.inf, 0)

        self._seed()
        self._reset()

    def _inspection_probability(self):
        prob = (self.inspection_rate_mean +
                self.inspection_rate_amplitude * np.cos(
                    (2 * np.pi / self.inspection_period) * self.t))
        return max(min(prob, 1), 0)

    def _disposal_cost(self):
        mean_cost = (
            self.market_price_mean + self.market_price_amplitude * np.sin(
                (2 * np.pi / self.market_period) * self.t))
        noise = scipy.stats.truncnorm.rvs(
            a=-2, b=2, scale=self.market_noise_scale)
        return mean_cost + noise

    def _step(self, action):
        if action == 0:  # Store
            cost = self.storage_cost * self.stored_waste
        elif action == 1:  # Dispose
            cost = self.disposal_cost * self.stored_waste
            self.stored_waste = 0
        elif action == 2:  # Dump
            cost = 0
            self.pollution += self.stored_waste
            self.stored_waste = 0
        else:
            raise ValueError('Invalid Action: {}'.format(action))

        inspection_probability = self._inspection_probability()
        self.inspection = self.np_random.binomial(1, inspection_probability)
        self.penalize = (self.inspection
                         and self.pollution >= self.pollution_threshold)
        reward = -cost
        if self.penalize:
            reward -= self.pollution_penalty

        self.t += 1
        self.stored_waste += 1
        self.pollution *= self.pollution_decay
        self.disposal_cost = self._disposal_cost()
        return self._get_observations(), reward, False, {
            'pollution': self.pollution,
            'inspection_probability': inspection_probability
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.t = 0
        self.stored_waste = 1
        self.pollution = 0
        self.disposal_cost = self._disposal_cost()
        self.inspection = False
        self.penalize = False
        return self._get_observations()

    def _get_observations(self):
        return (self.stored_waste, self.disposal_cost, int(self.inspection),
                int(self.penalize))

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        outfile.write((' '.join((
            'SW: {stored_waste:8d}',
            'DC: {disposal_cost:5.2f}',
            '{inspection}',
            '{penalize}',
            'P: {pollution:8.2f}', )) + '\n').format(
                stored_waste=self.stored_waste,
                disposal_cost=self.disposal_cost,
                inspection='I' if self.inspection else '_',
                penalize='P' if self.penalize else '_',
                pollution=self.pollution, ))
        if mode != 'human':
            return outfile

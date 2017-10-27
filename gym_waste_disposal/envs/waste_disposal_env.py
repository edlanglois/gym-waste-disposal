"""Waste Disposal Environment"""
import io
import sys

import gym
import numpy as np
import scipy.stats
from gym import spaces


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
        self.inspection_rate_amplitude = 1

        self.pollution_decay = 0.9
        self.pollution_threshold = 0.5
        self.pollution_penalty = 5000

        self.action_space = gym.spaces.Discrete(3)
        # yapf: disable
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.inf),
            spaces.Box(low=(self.market_price_mean
                            - self.market_price_amplitude
                            - 2 * self.market_noise_scale),
                       high=(self.market_price_mean
                             + self.market_price_amplitude
                             + 2 * self.market_noise_scale)),
            spaces.Discrete(2),
            spaces.Discrete(2)
        ))
        # yapf: enable
        self.reward_range(-np.inf, 0)

    def _inspection_probability(self):
        return (self.inspection_rate_mean +
                self.inspection_rate_amplitude * np.cos(
                    (2 * np.pi / self.inspection_period) * self.t))

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

        self.inspection = self.np_random.binomial(
            1, self._inspection_probability(self.t))
        self.penalize = (self.inspection
                         and self.pollution >= self.pollution_threshold)
        reward = -cost
        if self.penalize:
            reward -= self.pollution_penalty

        self.t += 1
        self.stored_waste += 1
        self.pollution *= self.pollution_decay
        return self._get_observations(), reward, False, {}

    def _reset(self):
        self.t = 0
        self.stored_waste = 1
        self.pollution = 0
        self.disposal_cost = self.disposal_cost()
        self.inspection = False
        self.penalize = False
        return self._get_observations(False, False)

    def _get_observations(self, inspection, penalize):
        return (self.stored_waste, self.disposal_cost, int(self.inspection),
                int(self.penalize))

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(
            ' '.join(('SW: {stored_waste:8d}', 'DC: {disposal_cost:5.2f}',
                      '{inspection}', '{penalize}')).format(
                          stored_waste=self.stored_waste,
                          disposal_cost=self.disposal_cost,
                          inspection='I' if self.inspection else '_',
                          penalize='P' if self.penalize else '_'))
        if mode != 'human':
            return outfile

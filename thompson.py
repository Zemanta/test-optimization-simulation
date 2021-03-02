from random import random
from typing import List, Dict
from collections import Counter

import numpy as np
import pandas as pd

from bandits import BetaBandit


class WeightedChoiceFailed(Exception):
    def __init__(self, relative_frequencies: List[float], message: str='Weighted choice failed:'):
        self.relative_frequencies = relative_frequencies
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.relative_frequencies} -> {self.message}'


class ThompsonSampling:
    def __init__(self, sample_size: int=1000, batch_size: int=1000):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.bandits = list()
        self.relative_frequencies = list()
    
    def add_bandit(self, positive_examples: int=0, negative_examples: int=0, alpha_prior: float=1., beta_prior: float=1.):
        self.bandits.append(BetaBandit(positive_examples, negative_examples, alpha_prior, beta_prior))

    def update_bandit(self, idx: int, positive_examples: int=0, negative_examples: int=0):
        self.bandits[idx].update(positive_examples, negative_examples)

    def generate_relative_frequencies(self):
        bandit_samples = dict()
        self.relative_frequencies = [0] * len(self.bandits)
        for i in range(len(self.bandits)):
            bandit = self.bandits[i]
            bandit_samples[i] = bandit.sample(self.sample_size)
        bandit_df = pd.DataFrame(bandit_samples)
        for i in range(bandit_df.shape[0]):
            self.relative_frequencies[bandit_df.iloc[i].idxmax()] += 1. / self.sample_size
    
    def weighted_choice(self) -> int:
        r = random()

        for i in range(len(self.relative_frequencies)):
            r -= self.relative_frequencies[i]
            if r < 0:
                return i

        raise WeightedChoiceFailed(self.relative_frequencies)

    def bandit_batch(self) -> Dict[int, int]:
        self.generate_relative_frequencies()
        strategy = [self.weighted_choice() for _ in range(self.batch_size)]
        counter = Counter(strategy)
        return dict(counter)


class RunThompsonSampling:
    def __init__(self, bandit_returns: List[float], alpha_priors: List[float]=None, beta_priors: List[float]=None, sample_size: int=1000, batch_size: int=1000, batches: int=10, simulations: int=2):
        self.bandit_returns = bandit_returns
        self.n_bandits = len(bandit_returns)
        self.bandits = list(range(self.n_bandits))

        self.sample_size = sample_size
        self.batch_size = batch_size
        self.batches = batches
        self.simulations = simulations

        self.df_bids = pd.DataFrame(columns=self.bandit_returns)
        self.df_clicks = pd.DataFrame(columns=self.bandit_returns)

        if alpha_priors is None:
            alpha_priors = [1.] * self.n_bandits
        if beta_priors is None:
            beta_priors = [1.] * self.n_bandits
    
    def init_bandits(self):
        self.bandit_positive_examples = [0] * self.n_bandits
        self.bandit_total_examples = [0] * self.n_bandits
        self.thomsam = ThompsonSampling(self.sample_size, self.batch_size)
        for i in self.bandits:
            self.thomsam.add_bandit()

    def run(self):
        for j in range(self.simulations):
            self.init_bandits()
            for i in range(self.batches):
                for key, val in self.thomsam.bandit_batch().items():
                    self.bandit_total_examples[key] += val
                    self.bandit_positive_examples[key] += np.random.binomial(val, self.bandit_returns[key])
                    self.thomsam.update_bandit(key, self.bandit_positive_examples[key], self.bandit_total_examples[key]- self.bandit_positive_examples[key])
                if self.df_bids.shape[0] < self.batches:
                        self.df_bids.loc[i] = self.bandit_total_examples
                        self.df_clicks.loc[i] = self.bandit_positive_examples
                else:
                    self.df_bids.loc[i] += self.bandit_total_examples
                    self.df_clicks.loc[i] += self.bandit_positive_examples
        self.df_bids /= self.simulations
        self.df_clicks /= self.simulations

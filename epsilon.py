import math
from typing import List

import numpy as np
import pandas as pd

from bandits import EpsilonBandit


class EpsilonGreedy:
    def __init__(self, epsilon: float=0.2, batch_size: int=1000):
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.bandits = list()
        self.best_bandits = list()
    
    def add_bandit(self, positive_examples: int=0):
        self.bandits.append(EpsilonBandit(positive_examples))

    def update_bandit(self, idx: int, positive_examples: int=0):
        self.bandits[idx].update(positive_examples)

    def get_best_bandit(self):
        idx = 0
        maxVal = 0
        for i in range(len(self.bandits)):
            val = self.bandits[i].get_value()
            if val > maxVal:
                maxVal = val
                idx = i
        self.best_bandits.append(idx)

    def bandit_batch(self) -> (int, int):
        self.get_best_bandit()
        n_bandits = len(self.bandits)
        exploration_total = self.batch_size * self.epsilon
        exploration = int(exploration_total / n_bandits)

        return exploration, self.best_bandits[-1]


class RunEpsilonGreedy:
    def __init__(self, bandit_returns: List[float], epsilon: float=0.2, batch_size: int=10000, batches: int=10, simulations: int=100):
        self.bandit_returns = bandit_returns
        self.n_bandits = len(bandit_returns)
        self.bandits = list(range(self.n_bandits))

        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batches = batches
        self.simulations = simulations

        self.df_bids = pd.DataFrame(columns=self.bandits)
        self.df_clicks = pd.DataFrame(columns=self.bandits)

    def init_bandits(self):
        self.first_batch = True
        self.bandit_positive_examples = [0] * self.n_bandits
        self.bandit_total_examples = [0] * self.n_bandits
        self.eps = EpsilonGreedy(self.epsilon, self.batch_size)
        for i in self.bandits:
            self.eps.add_bandit()
    
    def run(self):
        for j in range(self.simulations):
            self.init_bandits()
            for i in range(self.batches):
                exploration_examples, best_bandit = self.eps.bandit_batch()
                if self.first_batch:
                    self.first_batch = False
                    exploration_examples = self.batch_size // self.n_bandits
                for idx in self.bandits:
                    self.bandit_total_examples[idx] += exploration_examples
                    self.bandit_positive_examples[idx] += np.random.binomial(exploration_examples, self.bandit_returns[idx])
                    self.eps.update_bandit(idx, self.bandit_positive_examples[idx])
                
                exploitation_examples = self.batch_size - exploration_examples * self.n_bandits
                self.bandit_total_examples[best_bandit] += exploitation_examples
                self.bandit_positive_examples[best_bandit] += np.random.binomial(exploitation_examples, self.bandit_returns[best_bandit])

                if self.df_bids.shape[0] < self.batches:
                    self.df_bids.loc[i] = self.bandit_total_examples
                    self.df_clicks.loc[i] = self.bandit_positive_examples
                else:
                    self.df_bids.loc[i] += self.bandit_total_examples
                    self.df_clicks.loc[i] += self.bandit_positive_examples
        self.df_bids /= self.simulations
        self.df_clicks /= self.simulations

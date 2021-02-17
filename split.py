from typing import List

import numpy as np
import pandas as pd


class RunSplitTest:
    def __init__(self, bandit_returns: List[float], batch_size: int=1000, batches: int=10):
        self.bandit_returns = bandit_returns
        self.n_bandits = len(bandit_returns)
        self.bandits = list(range(self.n_bandits))
        self.bandit_positive_examples = [0] * self.n_bandits
        self.bandit_total_examples = [0] * self.n_bandits

        self.batch_size = batch_size
        self.batches = batches

        self.df_bids = pd.DataFrame(columns=self.bandits)
        self.df_clicks = pd.DataFrame(columns=self.bandits)
        
    def run(self):
        for i in range(self.batches):
            examples = self.batch_size // self.n_bandits
            for idx in self.bandits:
                self.bandit_total_examples[idx] += examples
                self.bandit_positive_examples[idx] += np.random.binomial(examples, self.bandit_returns[idx])
            self.df_bids.loc[i] = self.bandit_total_examples
            self.df_clicks.loc[i] = self.bandit_positive_examples

import math

import numpy as np


class BetaBandit:
    def __init__(self, alpha: int=0, beta: int=0, alpha_prior: float=1., beta_prior: float=1.):
        self.alpha = alpha + alpha_prior
        self.beta = beta + beta_prior
    
    def update(self, positive_examples: int=0, negative_examples: int=0):
        self.alpha += positive_examples
        self.beta += negative_examples
    
    def sample(self, n: int) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta, n)


class EpsilonBandit:
    def __init__(self, positive_examples: int=0):
        self.positive_examples = positive_examples
    
    def update(self, positive_examples: int=0):
        self.positive_examples += positive_examples
    
    def get_value(self) -> int:
        return self.positive_examples

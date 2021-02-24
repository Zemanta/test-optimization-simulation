import math
from typing import List

import pandas as pd
from scipy import stats

from split import RunSplitTest
from epsilon import RunEpsilonGreedy
from thompson import RunThompsonSampling
from plotting import plot_stacked_plots, plot_gain

def z_calc(p1: float, p2: float, n1: int, n2: int):
    p_star = (p1*n1 + p2*n2) / (n1 + n2)
    return (p2 - p1) / math.sqrt(p_star*(1 - p_star)*((1.0 / n1) + (1.0 / n2)))


def sample_required(p1: float, p2: float, alpha: float=0.01):
    n = 1
    while True:
        z = z_calc(p1, p2, n1=n, n2=n)
        p = 1 - stats.norm.cdf(z)
        if p < alpha:
            break
        n += 1
    return n


def closest_pair(bandits: List[float]) -> (float, float):
    bandits.sort()
    min_diff = float("inf")
    p1 = -1
    p2 = -1
    for i in range(len(bandits)-1):
        if bandits[i+1] - bandits[i] < min_diff:
            p1 = bandits[i]
            p2 = bandits[i+1]
    return p1, p2


def get_minimum_sample(bandits: List[float], alpha: float=0.01) -> int:
    p1, p2 = closest_pair(bandits)
    return sample_required(p1, p2, alpha)


def define_batches(examples_needed: int, batch_size: int):
    return math.ceil(examples_needed / batch_size)


def simulate(bandits: List[float], alpha: float=0.001, batch_size: int=1000, simulations: int=1000, epsilon: float=0.1, sample_size: int=1000) -> (RunSplitTest, RunEpsilonGreedy, RunThompsonSampling):
    examples_needed = get_minimum_sample(bandits, alpha)
    batches = define_batches(examples_needed, batch_size)

    rst = RunSplitTest(bandits,
                       batch_size=batch_size,
                       batches=batches,
                       simulations=simulations)

    reg = RunEpsilonGreedy(bandits,
                           epsilon=epsilon, 
                           batch_size=batch_size,
                           batches=batches,
                           simulations=simulations)

    rts = RunThompsonSampling(bandits,
                              alpha_priors=None,
                              beta_priors=None,
                              sample_size=sample_size,
                              batch_size=batch_size,
                              batches=batches,
                              simulations=simulations)
    
    rst.run()
    reg.run()
    rts.run()
    return rst, reg, rts


def run_simulations(bandits: List[float], alpha: float=0.001, batch_size: int=1000, simulations: int=1000, epsilon: float=0.1, sample_size: int=1000):
    rst, reg, rts = simulate(bandits=bandits,
                             alpha=alpha,
                             batch_size=batch_size,
                             simulations=simulations,
                             epsilon=epsilon,
                             sample_size=sample_size)

    plot_stacked_plots(rst=rst,
                       reg=reg,
                       rts=rts)

    plot_gain(rst=rst,
              reg=reg,
              rts=rts)

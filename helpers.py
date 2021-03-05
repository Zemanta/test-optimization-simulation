import math
from typing import List

import pandas as pd
from scipy import stats

from split import RunSplitTest
from epsilon import RunEpsilonGreedy
from thompson import RunThompsonSampling
from plotting import plot_stacked_plots, plot_line_plots, plot_gain, stacked_plot

def z_calc(p1: float, p2: float, n1: int, n2: int) -> float:
    """
    Calculates the z value for the difference of two sample means.

    Args:
        p1: Mean of first sample.
        p2: Mean of second sample.
        n1: Sample size of the first sample.
        n2: Sample size of the second sample.
    Returns:
        The calculated z value.
    """
    p_star = (p1*n1 + p2*n2) / (n1 + n2)
    return (p2 - p1) / math.sqrt(p_star*(1 - p_star)*((1.0 / n1) + (1.0 / n2)))


def sample_required(p1: float, p2: float, alpha: float=0.01) -> int:
    """
    Calculates the sample size needed to provide a test power of (1-alpha)
    in which we are testing h0: p2-p1==0.

    Args:
        p1: Mean of first sample.
        p2: Mean of second sample.
        alpha: Type one error.
    Returns:
        The calculated sample size.
    """
    n = 1
    while True:
        z = z_calc(p1, p2, n1=n, n2=n)
        p = 1 - stats.norm.cdf(z)
        if p < alpha:
            break
        n += 1
    return n


def closest_pair(bandits: List[float]) -> (float, float):
    """
    Finds the two bandits that have the closest average return
    and returns their returns.

    Args:
        bandits: list of average bandit returns.
    Returns:
        The average return of the two bandits that are the most
        similar.
    """
    bandits.sort()
    min_diff = float("inf")
    p1 = -1
    p2 = -1
    for i in range(len(bandits)-1):
        if bandits[i+1] - bandits[i] < min_diff:
            p1 = bandits[i]
            p2 = bandits[i+1]
            min_diff = p2 - p1
    return p1, p2


def get_minimum_sample(bandits: List[float], alpha: float=0.01) -> int:
    """
    Gets the minimum sample size required to provide a test power of
    (1-alpha/len(bandits)), this includes the p-value Bonferroni correction.

    Args:
        bandits: list of average bandit returns.
        alpha: Type one error.
    Returns:
        Needed sample size.
    """
    p1, p2 = closest_pair(bandits)
    return sample_required(p1, p2, alpha/len(bandits))


def get_number_batches(examples_needed: int, batch_size: int) -> int:
    """
    Gets the number of batches based on the number of needed examples
    and provided batch size.

    Args:
        examples_needed: Total number of examples needed.
        batch_size: Number of examples per batch.
    Returns:
        The number of batches.
    """
    return math.ceil(examples_needed / batch_size)


def simulate(bandits: List[float], alpha: float=0.001, batch_size: int=5000, simulations: int=1000, epsilon: float=0.1, sample_size: int=1000) -> (RunSplitTest, RunEpsilonGreedy, RunThompsonSampling):
    """
    Runs simulations for split tests, Epsilon-greedy multi-armed bandits
    and Thompson sampling based on the provided parameters.

    Args:
        bandits: list of average bandit returns.
        alpha: Type one error.
        batch_size: Number of examples per batch.
        simulations: Number of simaltions per test type.
        epsilon: percentage of exploration in epsilon-greedy MAB
        sample_size: sample size per bandit for each Thompson sampling batch
    Returns:
        The classes for each type of test.
    """
    examples_needed = get_minimum_sample(bandits, alpha)
    batches = get_number_batches(examples_needed, batch_size)

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


def run_simulations(bandits: List[float], alpha: float=0.001, batch_size: int=1000,
                    simulations: int=1000, epsilon: float=0.1, sample_size: int=1000):
    """
    Starts the simulation process, gets the results and makes plots.

    Args:
        bandits: list of average bandit returns.
        alpha: Type one error.
        batch_size: Number of examples per batch.
        simulations: Number of simaltions per test type.
        epsilon: percentage of exploration in epsilon-greedy MAB
        sample_size: sample size per bandit for each Thompson sampling batch
    """
    rst, reg, rts = simulate(bandits=bandits,
                             alpha=alpha,
                             batch_size=batch_size,
                             simulations=simulations,
                             epsilon=epsilon,
                             sample_size=sample_size)

    plot_stacked_plots(rst=rst,
                       reg=reg,
                       rts=rts)

    plot_line_plots(rst=rst,
                    reg=reg,
                    rts=rts)

    plot_gain(rst=rst,
              reg=reg,
              rts=rts)

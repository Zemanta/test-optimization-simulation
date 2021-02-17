import math
from typing import List

from scipy import stats


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

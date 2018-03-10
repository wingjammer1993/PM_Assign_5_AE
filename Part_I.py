from scipy.stats import gamma
from scipy.stats import norm
import math
import numpy as np


# I = 1, M = 2, U = 3, S = 4
def give_likelihood_weighting(num_samples):
    # Fix evidence variables
    samples = {}
    for i in range(0, num_samples):
        intell = 100 + np.random.randn()*15
        major = np.random.rand() < 1/(1 + math.exp(-(intell - 110)/5))
        uni = np.random.rand() < 1/(1 + math.exp(-(intell - 110)/5))
        salary = np.random.gamma(0.1*intell + major + 3*uni, 5)
        weight = gamma.pdf(salary, 0.1*intell + major + 3*uni, 5)
        samples[(intell, major, uni, weight)] = weight
    return samples


if __name__ == "__main__":
    num = 10000
    dict_sample = give_likelihood_weighting(num)
    print(dict_sample)


from scipy.stats import gamma
import math
import numpy as np


# I = 1, M = 2, U = 3, S = 4
def give_likelihood_weighting(num_samples, salary_weight):
    # Fix evidence variables
    samples = {}
    for i in range(0, num_samples):
        intell = 100 + np.random.randn()*15
        major = np.random.rand() < 1/(1 + math.exp(-(intell - 110)/5))
        uni = np.random.rand() < 1/(1 + math.exp(-(intell - 110)/5))
        weight = gamma.pdf(salary_weight, 0.1*intell + major + 3*uni, 5)
        samples[(intell, major, uni, 120)] = weight
    return samples


if __name__ == "__main__":
    num = 10000
    dict_sample = give_likelihood_weighting(num, 120)
    print(dict_sample)


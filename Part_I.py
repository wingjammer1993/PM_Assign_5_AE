from scipy.stats import gamma
from scipy.stats import norm
import math
import numpy as np


# I = 1, M = 2, U = 3, S = 4
def give_likelihood_weighting(num_samples):
    # Fix evidence variables
    samples = {}
    for i in range(0, num_samples):
        a = 1
        b = 1
        c = 1
        d = 1
        variables = {'I': np.nan, 'M': np.nan, 'U': np.nan, 'S': np.nan}
        evidence_idx = np.random.choice([1, 4], 2)
        if 1 in evidence_idx:
            variables['I'] = np.random.rand()*(130 - 70) + 70
            a = norm.pdf(variables['I'], 100, 15)
        if 2 in evidence_idx:
            variables['M'] = np.random.choice([0, 1])
            if np.isnan(variables['I']):
                variables['I'] = np.random.rand() * (130 - 70) + 70
            b = 1 / (1 + math.exp(-(variables['I'] - 110) / 5))
            if variables['M'] != 1:
                b = 1 - b
        if 3 in evidence_idx:
            variables['U'] = np.random.choice([0, 1])
            if np.isnan(variables['I']):
                variables['I'] = np.random.rand() * (130 - 70) + 70
            c = 1 / (1 + math.exp(-(variables['I'] - 110) / 5))
            if variables['U'] != 1:
                c = 1 - c
        if 4 in evidence_idx:
            variables['S'] = np.random.gamma(12, 5)
            if np.isnan(variables['I']):
                variables['I'] = np.random.rand() * (130 - 70) + 70
            if np.isnan(variables['M']):
                variables['M'] = np.random.choice([0, 1])
            if np.isnan(variables['U']):
                variables['U'] = np.random.choice([0, 1])
            d = gamma.pdf(variables['S']*0.1 + variables['M'] + 3*variables['U'], 5)

        if np.isnan(variables['I']):
            variables['I'] = np.random.rand() * (130 - 70) + 70
        if np.isnan(variables['M']):
            variables['M'] = np.random.choice([0, 1])
        if np.isnan(variables['U']):
            variables['U'] = np.random.choice([0, 1])
        if np.isnan(variables['S']):
            variables['S'] = np.random.gamma(12, 5)

        if (variables['I'], variables['M'], variables['U'], variables['S']) in samples:
            samples[(variables['I'], variables['M'], variables['U'], variables['S'])] += a*b*c*d
        else:
            samples[(variables['I'], variables['M'], variables['U'], variables['S'])] = a*b*c*d

    return samples


if __name__ == "__main__":
    num = 10000
    dict_sample = give_likelihood_weighting(num)
    print(dict_sample)


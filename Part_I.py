from scipy.stats import gamma
import math
import numpy as np


def give_likelihood_weighting(num_samples, salary_weight):
    samples = {}
    for i in range(0, num_samples):
        intell = 100 + np.random.randn()*15
        major = np.random.rand() < 1/(1 + math.exp(-(intell - 110)/5))
        uni = np.random.rand() < 1/(1 + math.exp(-(intell - 100)/5))
        weight = gamma.pdf(salary_weight, 0.1*intell + major + 3*uni, 5)
        samples[(intell, major, uni, salary_weight)] = weight
    return samples


def compute_joint_posterior(likelihood_samples):
    joint_post = {('cs', 'met'): 0, ('cs', 'uc'): 0, ('biz', 'met'): 0, ('biz', 'uc'): 0}
    for sample in likelihood_samples:
        if sample[1]:
            if sample[2]:
                joint_post[('cs', 'uc')] += likelihood_samples[sample]
            else:
                joint_post[('cs', 'met')] += likelihood_samples[sample]
        else:
            if sample[2]:
                joint_post[('biz', 'uc')] += likelihood_samples[sample]
            else:
                joint_post[('biz', 'met')] += likelihood_samples[sample]
    joint_post = {k: v/sum(joint_post.values()) for k, v in joint_post.items()}
    return joint_post


if __name__ == "__main__":
    num = 10000
    dict_sample_1 = give_likelihood_weighting(num, 120)
    posterior_1 = compute_joint_posterior(dict_sample_1)
    print(posterior_1)

    dict_sample_2 = give_likelihood_weighting(num, 60)
    posterior_2 = compute_joint_posterior(dict_sample_2)
    print(posterior_2)

    dict_sample_3 = give_likelihood_weighting(num, 20)
    posterior_3 = compute_joint_posterior(dict_sample_3)
    print(posterior_3)


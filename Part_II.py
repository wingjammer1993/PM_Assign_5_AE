import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math


def x_given_y(y, means, std_devs):
    mean_xy = means[0] + (std_devs[0][-1]/std_devs[-1][-1])*(y - means[-1])
    std_dev_xy = std_devs[0][0] - (std_devs[0][-1]/std_devs[-1][-1])*std_devs[-1][0]
    return np.random.normal(mean_xy, std_dev_xy)


def y_given_x(x, means, std_devs):
    mean_yx = means[-1] + (std_devs[-1][0]/std_devs[0][0])*(x - means[0])
    std_dev_yx = std_devs[-1][-1] - (std_devs[-1][0]/std_devs[0][0])*std_devs[0][-1]
    return np.random.normal(mean_yx, std_dev_yx)


def gibbs_sampler(y_init, means, std_devs, num_samples):
    x_samples = []
    y_samples = []
    y = y_init
    for i in range(0, num_samples):
        x = x_given_y(y, means, std_devs)
        y = y_given_x(x, means, std_devs)
        x_samples.append(x)
        y_samples.append(y)
    return x_samples, y_samples


if __name__ == "__main__":
    samples = 100000
    y_in = 0
    mean = [1, 0]
    std_dev = np.array([[1, -0.5], [-0.5, 3]])
    dist_x, dist_y = gibbs_sampler(y_in, mean, std_dev, samples)
    ex = np.linspace(norm.ppf(0.00000000000000001), norm.ppf(0.999999999999999), 100)
    plt.hist(dist_x, 50, normed=True)
    plt.plot(ex, norm.pdf(ex, 1, 1), 'r-', lw=5, alpha=0.6, label='norm pdf')
    plt.show()
    plt.hist(dist_y, 50, normed=True)
    plt.plot(ex, norm.pdf(ex, 0, 3), 'r-', lw=5, alpha=0.6, label='norm pdf')
    plt.show()





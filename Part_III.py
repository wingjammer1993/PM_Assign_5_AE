import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def get_probability_first(sample):
    if sample < 0 or sample > 1:
        return 0
    else:
        return sample**3


def get_probability_second(conditional, sample):
    if sample < 0 or sample > 1:
        return 0
    else:
        return 1 - abs(sample - conditional)


def metropolis_hastings(num_samples):
    accepted_x = OrderedDict()
    accepted_y = OrderedDict()
    mean = 0.1
    dev = 0.3
    x_init, y_init = 0.9, 0.9
    count = 0
    for i in range(0, num_samples):
        x = np.random.normal(mean, dev)
        y = np.random.normal(mean, dev)
        x = x + x_init
        y = y + y_init
        p_1 = get_probability_first(x)
        p_2 = get_probability_second(x, y)
        p_3 = get_probability_first(x_init)
        p_4 = get_probability_second(x_init, y_init)
        if p_3 > 0 and p_4 > 0:
            acceptance = min(1, (p_1*p_2)/(p_3*p_4))
        else:
            acceptance = 1

        if acceptance > np.random.rand():
            x_init = x
            y_init = y
            accepted_x[count] = x
            accepted_y[count] = y
            count += 1

    return accepted_x, accepted_y


if __name__ == "__main__":
    num = 1000000
    accepted_1, accepted_2 = metropolis_hastings(num)
    plt.scatter(list(accepted_2.values()), list(accepted_1.values()))
    plt.show()
    fig = plt.figure()
    plt.xlabel('G')
    plt.ylabel('F')
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    k = ax.hist2d(list(accepted_2.values()), list(accepted_1.values()), bins=20, normed=True)
    #plt.colorbar(k[3], ax=ax)
    plt.show()
    plt.plot(list(accepted_1.keys())[0:1000], list(accepted_1.values())[0:1000])
    plt.plot(list(accepted_2.keys())[0:1000], list(accepted_2.values())[0:1000])
    plt.show()







import pymc3 as pm
import numpy as np
import theano

g1_prob = np.array([0.5, 0.5])

g2_g1_prob = np.array([[0.9, 0.1], [0.1, 0.9]])

g3_g1_prob = np.array([[0.9, 0.1], [0.1, 0.9]])

x2_mu, x2_sd = np.array([[50, 60],
                         [3.162, 3.162]])

x3_mu, x3_sd = np.array([[50, 60],
                         [3.162, 3.162]])


with pm.Model() as dna_model:

    g1 = pm.Categorical('g1', p=g1_prob)

    g2_prob = theano.shared(g2_g1_prob)  # make numpy-->theano

    g2_0 = g2_prob[g1]  # select the prob array that "happened" thanks to parents

    g2 = pm.Categorical('g2', p=g2_0)

    g3_prob = theano.shared(g3_g1_prob)  # make numpy-->theano

    g3_0 = g3_prob[g1]  # select the prob array that "happened" thanks to parents

    g3 = pm.Categorical('g3', p=g3_0)

    x2 = pm.Normal('x2', mu=50 + 10*g2, tau=3.162)

    x3 = pm.Normal('x3', mu=50 + 10*g3, tau=3.162)


with dna_model:
    trace = pm.sample(1)
    print(pm.summary(trace, varnames=['g1', 'g2', 'g3'], start=1))

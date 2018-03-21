import pymc3 as pm
import math
import matplotlib.pyplot as plt


with pm.Model() as dna_model:

    g1 = pm.Bernoulli('g1', .5)

    g2 = pm.Bernoulli('g2', pm.math.switch(g1 > 0, 0.9, 0.1))

    g3 = pm.Bernoulli('g3', pm.math.switch(g1 > 0, 0.9, 0.1))

    var_1 = pm.math.switch(g1 > 0, 50, 60)

    var_2 = pm.math.switch(g2 > 0, 50, 60)

    var_3 = pm.math.switch(g3 > 0, 50, 60)

    x1 = pm.Normal('x1', var_1, math.sqrt(10))

    x2 = pm.Normal('x2', var_2, math.sqrt(10),observed=50)

    x3 = pm.Normal('x3', var_3, math.sqrt(10))

with dna_model:

    step = pm.Metropolis()

    trace = pm.sample(10000, step=step)

    print(pm.summary(trace))

    pm.traceplot(trace)

    trace_g1 = trace['g1'][:]
    trace_g2 = trace['g2'][:]
    trace_g3 = trace['g3'][:]
    trace_x1 = trace['x1'][:]
    trace_x2 = trace['x2'][:]
    trace_x3 = trace['x3'][:]
    plt.show()

    plt.hist(trace_x1)
    plt.hist(trace_x2)
    plt.hist(trace_x3)
    plt.show()

    print(trace_g1, trace_g2, trace_g3, trace_x1, trace_x2, trace_x3)





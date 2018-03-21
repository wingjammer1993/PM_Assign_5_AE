import pymc3 as pm
import math
import matplotlib.pyplot as plt
import theano.tensor


with pm.Model() as dna_model:

    g1 = pm.Bernoulli('g1', .5)

    g2 = pm.Bernoulli('g2', pm.math.switch(g1 < 1, 0.1, 0.9))

    g3 = pm.Bernoulli('g3', pm.math.switch(g1 < 1, 0.1, 0.9))

    var_1 = pm.math.switch(g1 < 1, 50, 60)

    var_2 = pm.math.switch(g2 < 1, 50, 60)

    var_3 = pm.math.switch(g3 < 1, 50, 60)

    x1 = pm.Normal('x1', var_1, math.sqrt(10))

    x2 = pm.Normal('x2', var_2, math.sqrt(10), observed=50)

    x3 = pm.Normal('x3', var_3, math.sqrt(10))

with dna_model:

    step = pm.Metropolis()
    start = pm.find_MAP()
    trace = pm.sample(10000, step=step, start=start)

    print(pm.summary(trace))

    pm.traceplot(trace)
    plt.show()

    trace_g1 = trace['g1'][:].tolist()
    trace_g2 = trace['g2'][:].tolist()
    trace_g3 = trace['g3'][:].tolist()
    trace_x1 = trace['x1'][:].tolist()
    trace_x3 = trace['x3'][:].tolist()

    count = 0
    for i in trace_g1:
        if i == 1:
            count += 1

    probability_conditional = count/float(len(trace_g1))

    print(probability_conditional)





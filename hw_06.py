import pymc3 as pm
import numpy as np

observed_values = [1.]

with pm.Model() as dna_model:

    g1 = pm.Bernoulli('g1', .5)

    p_g2 = pm.Deterministic('p_g2', pm.math.switch(g1, 0.1, 0.9))

    g2 = pm.Bernoulli('g2', p_g2)

    p_g3 = pm.Deterministic('p_g3', pm.math.switch(g1, 0.1, 0.9))

    g3 = pm.Bernoulli('g3', p_g3)

    p_x2 = pm.Deterministic('p_x2', pm.math.switch(g2, pm.Normal(p_x2, 50, 3.1662), pm.Normal('p_x2', 60, 3.1662)))

    x2 = pm.Normal('x2', p_x2)

    p_x3 = pm.Deterministic('p_x3', pm.math.switch(g2, pm.Normal('p_x3', 50, 3.1662), pm.Normal('p_x3', 60, 3.1662)))

    x3 = pm.Normal('x3', p_x3)


mc_mc = pm.Metropolis(dna_model)
pm.sample(10000, 2000)

import pymc3 as pm
import math
import matplotlib.pyplot as plt

# Notation change note -
# Here, gi=0 indicates gi=1 as per problem statement convention and gi=1 indicates gi=2

with pm.Model() as dna_model:

    # Create a variable g1 with Bernoulli distribution
    # g1 = 0 with probability 0.5
    # g1 = 1 with probability 0.5
    g1 = pm.Bernoulli('g1', .5)

    # Create a variable g2 with Bernoulli distribution.
    # If condition is TRUE, g1 = 0, g2 = 1 with probability 0.1 and g2 = 0 with probability 0.9
    # If condition is FALSE, g1 = 1, g2 = 1 with probability 0.9 and g2 = 0 with probability 0.1
    g2 = pm.Bernoulli('g2', pm.math.switch(g1 < 1, 0.1, 0.9))

    # Create a variable g3 with Bernoulli distribution.
    # If condition is TRUE, g1 = 0, g3 = 1 with probability 0.1 and g3 = 0 with probability 0.9
    # If condition is FALSE, g1 = 1, g3 = 1 with probability 0.9 and g3 = 0 with probability 0.1
    g3 = pm.Bernoulli('g3', pm.math.switch(g1 < 1, 0.1, 0.9))

    # Create temporary variables var_1, var_2 and var_3 which depend on g1, g2, g3 respectively.
    # If g1 == 0, var_1 = 50, If g1 == 1, var_1 = 60
    var_1 = pm.math.switch(g1 < 1, 50, 60)
    # If g2 == 0, var_2 = 50, If g2 == 1, var_2 = 60
    var_2 = pm.math.switch(g2 < 1, 50, 60)
    # If g3 == 0, var_3 = 50, If g3 == 1, var_3 = 60
    var_3 = pm.math.switch(g3 < 1, 50, 60)

    # Create a variable x1 with Normal distribution
    # If g1 == 0, probability distribution of x1 is a gaussian with mean = 50
    # If g1 == 1, probability distribution of x1 is a gaussian with mean = 60
    x1 = pm.Normal('x1', var_1, math.sqrt(10))

    # Create a variable x2 with Normal distribution
    # x2 is an evidence variable, so it is not sampled and fixed at x2 = 50
    x2 = pm.Normal('x2', var_2, math.sqrt(10), observed=50)

    # Create a variable x3 with Normal distribution
    # If g3 == 0, probability distribution of x3 is a gaussian with mean = 50
    # If g3 == 1, probability distribution of x3 is a gaussian with mean = 60
    x3 = pm.Normal('x3', var_3, math.sqrt(10))

with dna_model:

    # Use the Metropolis sampling algorithm
    step = pm.Metropolis()
    # Find a good initialization for sampling
    start = pm.find_MAP()
    # Sample 100000 number of times
    trace = pm.sample(100000, step=step, start=start)
    # Trace the plot
    pm.traceplot(trace)
    # Show the summary of the samples
    print(pm.summary(trace))
    plt.show()

    # Part I
    trace_g1 = trace['g1'][:].tolist()
    # Find the conditional probability P(G1=1|X2=50)
    # P(G1=1|X2=50) =  P(G1=1,X2=50)/  P(X2=50)
    # P(G1=1|X2=50) =  P(G1=1,X2=50)/ [P(G1=0,X2=50) +  P(G1=1,X2=50)]
    # P(G1=1|X2=50) = #samples with G1 = 1 / #total samples
    count_1 = 0
    for i in trace_g1:
        if i == 1:
            count_1 += 1

    probability_conditional_1 = count_1/float(len(trace_g1))

    # Part II
    trace_x1 = trace['x3'][:].tolist()
    # Find the conditional probability P(X3=50|X2=50)
    # P(X3=50|X2=50) =  P(X3=50,X2=50)/ P(X2=50)
    # P(X3=50,X2=50) is approximated using P(49.5<X3<50.5,X2=50)
    # P(X3=50|X2=50) = #samples with 49.5<X3<50.5 / #total samples
    count_2 = 0
    for i in trace_x1:
        if 49.5 < i < 50.5:
            count_2 += 1

    probability_conditional_2 = count_2/float(len(trace_x1))

    print(probability_conditional_1)
    print(probability_conditional_2)





## Evolutionary Algorithm to Solve an N-Queens Problem

import numpy as np

# Parameters:
P = 5 # Population size
N = 8 # Number of queens

# Auxiliary functions:

def evaluate(individual):
     for i in individual:
         for j in individual[i]:


# Create population
# Random 0-1 matrices of size 8x8

array_seed = np.array([1] * N + [0] * (N * (N - 1)))
population = []
for i in range(P):
    np.random.shuffle(array_seed)
    population.append(array_seed.reshape((N, N)))



import pdb;pdb.set_trace()

## Evolutionary Algorithm to Solve an N-Queens Problem

import numpy as np

# Parameters:
P = 1 # Population size
N = 4 # Number of queens

# Auxiliary functions:

def evaluate(individual):
    conflicts = 0
    for i in range(N):
        for j in range(N):
            if individual[i][j] == 1:
                for p in range(N):
                    for q in range(N):
                        if (i != p or j != q) \
                            and ((i + q == p + j) \
                                or (i + j == p + q) \
                                or (i == p) or (j == q)):
                            if individual[p][q] == 1:
                                conflicts += 1
    return conflicts // 2



# Create population
# Random 0-1 matrices of size 8x8

array_seed = np.array([1] * N + [0] * (N * (N - 1)))
population = []
conflicts = []
for i in range(P):
    np.random.shuffle(array_seed)
    individual = array_seed.reshape((N, N))
    population.append(individual)
    conflicts.append(evaluate(individual))


import pdb;pdb.set_trace()

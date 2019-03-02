## Evolutionary Algorithm to Solve an N-Queens Problem
import numpy as np
import random

# Parameters:
P = 11 # Population size
N = 4 # Number of queens
S = .4 # Proportion of population that is selected each generation
pr_b = 1 # Breeding probability
pr_m = .1 # Mutation probability

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

def breed(individual1, individual2):
    individual1 = individual1.reshape(N * N)
    individual2 = individual2.reshape(N * N)
    halfway = (N * N) // 2
    if np.random.uniform() < pr_b:
        offspring1 = np.concatenate([individual1[:halfway], individual2[halfway:]])
        offspring2 = np.concatenate([individual2[:halfway], individual1[halfway:]])
        offspring1 = offspring1.reshape((N,N))
        offspring2 = offspring2.reshape((N,N))
        return offspring1, offspring2
    else:
        return individual1, individual2

# Create population
# Random 0-1 matrices of size 8x8
array_seed = np.array([1] * N + [0] * (N * (N - 1)))
population = []
conflicts = []
for i in range(P):
    np.random.shuffle(array_seed)
    individual = np.copy(array_seed.reshape((N, N)))
    population.append(individual)
    conflicts.append(evaluate(individual))


# Assign probabilities of surival:
fitness = []
total_conflicts = sum(conflicts)
for conflict in conflicts:
    fitness.append((1 - conflict / total_conflicts) / (P - 1))


# Select surviving population:
total_population_data = list(zip(population, conflicts, fitness))
survivors = []
while len(survivors) < int(S * P) or len(survivors) % 2 != 0:
    potential_survivor = random.choice(total_population_data)
    if np.random.uniform() < potential_survivor[2]:
        survivors.append(potential_survivor[0])

# Cross-over:
half_survivors = len(survivors) // 2
first_half = survivors[:half_survivors]
second_half = survivors[half_survivors:]
paired_survivors = list(zip(first_half, second_half))

offspring = []

for pair in paired_survivors:
    offspring1, offspring2 = breed(pair[0], pair[1])
    offspring.append(offspring1)
    offspring.append(offspring2)

# Repair offspring:
for individual in offspring:
    while np.count_nonzero(individual) > N:
        ranom_one_index = random.choice(np.nonzero(individual.reshape(N*N)))

import pdb;pdb.set_trace()

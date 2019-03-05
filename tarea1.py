#### Evolutionary Algorithm to Solve an N-Queens Problem
import numpy as np
import random

# Global parameters:
runs = 30
generations = 1000

### Matrix Solution

## Parameters:
P = 30 # Population size
N = 8 # Number of queens
S = .4 # Proportion of population that is selected each generation
pr_b = 1 # Breeding probability
pr_m = 1 # Mutation probability

## Auxiliary functions:
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

def sort_attacks(val):
    return val[1]

## Define function to run one generation:
def run_one_generation_matrix(population):
    # Evaluate
    conflicts = []
    for individual in population:
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
        reshaped_individual = individual.reshape(N*N)
        while np.count_nonzero(reshaped_individual) != N:
            if np.count_nonzero(reshaped_individual) > N:
                random_one_index = random.choice(np.nonzero(reshaped_individual)[0])
                reshaped_individual[random_one_index] = 0
            if np.count_nonzero(reshaped_individual) < N:
                random_zero_index = random.choice(np.where(reshaped_individual == 0)[0])
                reshaped_individual[random_zero_index] = 1
        individual = reshaped_individual.reshape((N, N))

    # Mutate:
    for individual in offspring:
        if np.random.uniform() < pr_m:
            reshaped_individual = individual.reshape(N*N)
            random_one_index = random.choice(np.nonzero(reshaped_individual)[0])
            random_zero_index = random.choice(np.where(reshaped_individual == 0)[0])
            reshaped_individual[random_one_index] = 0
            reshaped_individual[random_zero_index] = 1
            individual = reshaped_individual.reshape((N, N))

    # Replace:
    # Sort original population by number of attacks and replace bottom
    # population by new offspring.

    total_population_data.sort(key = sort_attacks)
    new_population = list(list(zip(*total_population_data))[0])
    number_of_offspring = len(offspring)
    new_population = new_population[:(P - number_of_offspring)]
    new_population = new_population + offspring

    population = new_population

    # Report final evaluation:
    conflicts = []
    for individual in population:
        conflicts.append(evaluate(individual))
    fitness = []
    total_conflicts = sum(conflicts)
    for conflict in conflicts:
        fitness.append((1 - conflict / total_conflicts) / (P - 1))
    total_population_data = list(zip(population, conflicts, fitness))
    total_population_data.sort(key = sort_attacks)

    return total_population_data

## Main matrix solution:

def run_matrix_evolution():
    ## Create initial population
    # Random 0-1 matrices of size NxN
    array_seed = np.array([1] * N + [0] * (N * (N - 1)))
    population = []
    for i in range(P):
        np.random.shuffle(array_seed)
        individual = np.copy(array_seed.reshape((N, N)))
        population.append(individual)

    for gen in range(generations):
        total_population_data = run_one_generation_matrix(population)
        best_configuration = total_population_data[0][0]
        conflicts = total_population_data[0][1]
        if conflicts == 0:
            break
    return gen, best_configuration, conflicts


total_report = []
for run in range(runs):
    gen, best_configuration, conflicts = run_matrix_evolution()
    total_report.append((gen, best_configuration, conflicts))

import pdb;pdb.set_trace()

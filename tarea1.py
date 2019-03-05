#### Evolutionary Algorithm to Solve an N-Queens Problem
import numpy as np
import random
import math
from collections import Counter

### Permutation Solution

## Parameters:
runs = 30
generations = 100
P = 100 # Population size
N = 8 # Number of queens
S = 5 # Number of randomly selected ind. for selection.
pr_b = 1 # Breeding probability
pr_m = .8 # Mutation probability

## Auxiliary functions:

def convert(individual):
    matrix = np.array([0] * N * N)
    matrix = matrix.reshape((N, N))
    for i in range(len(individual)):
        matrix[i][individual[i]] = 1
    return matrix

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

def sort_attacks(val):
    return val[1]

def run_one_generation_permutation(population):
    # Evaluate
    conflicts = []
    for individual in population:
        matrix = convert(individual)
        conflicts.append(evaluate(matrix))

    # Order by value:
    evaluated_population = list(zip(population, conflicts))

    # Select S random:
    random_selection = []
    while len(random_selection) < S:
        random_selection.append(random.choice(evaluated_population))

    # Sort and get best two:
    random_selection.sort(key = sort_attacks)
    father = random_selection[0][0]
    mother = random_selection[1][0]

    halfway = N // 2
    if np.random.uniform() < pr_b:
        offspring1 = np.concatenate((father[:halfway], mother[halfway:]))
        offspring2 = np.concatenate((mother[:halfway], father[halfway:]))

    # Repair:
    offspring_list = [offspring1, offspring2]
    offspring_conflicts = []
    for offspring in offspring_list:
        while len(np.unique(offspring)) != len(offspring):
            repeated = [item for item, count in Counter(offspring).items() if count > 1]
            for repeated_value in repeated:
                random_repeated_index = random.choice(np.where(offspring == repeated_value)[0])
                for i in range(N):
                    if not i in offspring:
                        offspring[random_repeated_index] = i
        # Mutate:
        if np.random.uniform() < pr_m:
            random_element = random.choice(offspring)
            random_substitute = random.choice(offspring)
            while random_element == random_substitute:
                random_substitute = random.choice(offspring)
            random_element_index = np.where(offspring == random_element)[0]
            random_substitute_index = np.where(offspring == random_substitute)[0]
            offspring[random_element_index] = random_substitute
            offspring[random_substitute_index] = random_element
        # Evaluate
        matrix = convert(offspring)
        offspring_conflicts.append(evaluate(matrix))


    # Replace:
    # Evaluate offspring:

    population = population + offspring
    conflicts = conflicts + offspring_conflicts
    evaluated_population = list(zip(population, conflicts))
    evaluated_population.sort(key = sort_attacks)
    evaluated_population = evaluated_population[:P]

    # Report final results:
    total_population_data = evaluated_population

    return total_population_data


def run_permutation_evolution():
    ## Create initial population
    # Random 0-7 arrays of size 1xN
    array_seed = np.arange(N)
    population = []
    for i in range(P):
        np.random.shuffle(array_seed)
        individual = np.copy(array_seed)
        population.append(individual)

    for gen in range(generations):
        total_population_data = run_one_generation_permutation(population)
        best_configuration = total_population_data[0][0]
        conflicts = total_population_data[0][1]
        if conflicts == 0:
            break
    return gen, best_configuration, conflicts

total_report = []
for run in range(runs):
    gen, best_configuration, conflicts = run_permutation_evolution()
    total_report.append((gen, best_configuration, conflicts))


import pdb;pdb.set_trace()

### Matrix Solution

## Parameters:
runs = 30
generations = 1000
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
    if np.random.uniform() < pr_b:
        individual1 = individual1.reshape(N * N)
        individual2 = individual2.reshape(N * N)
        halfway = (N * N) // 2
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

    # Order by value:
    evaluated_population = list(zip(population, conflicts))
    evaluated_population.sort(key = sort_attacks)
    population = list(list(zip(*evaluated_population))[0])
    conflicts = list(list(zip(*evaluated_population))[1])
    # Assign probabilities of surival:
    fitness = []
    total_conflicts = sum(conflicts)
    for i in range(len(conflicts)):
        fitness.append((1 - sum(conflicts[:i]) / total_conflicts))


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
        if np.array_equal(pair[0],pair[1])\
         or np.array_equal(offspring1, pair[0])\
          or np.array_equal(offspring2, pair[1]):
            offspring.append(offspring1)
            offspring.append(offspring2)

    # Repair offspring:
    for individual in offspring:
        reshaped_individual = individual.reshape(N*N)
        while np.count_nonzero(reshaped_individual) != N:
            if np.count_nonzero(reshaped_individual) > N:
                random_one_index = random.choice(np.nonzero(reshaped_individual)[0])
                reshaped_individual[random_one_index] = 0
            else:
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
    # Join offspring, evaluate and remove the worst performing individuals

    all_population = population + offspring
    conflicts = []
    for individual in all_population:
        conflicts.append(evaluate(individual))
    evaluated_population = list(zip(all_population, conflicts))
    evaluated_population.sort(key = sort_attacks)
    population = list(list(zip(*evaluated_population))[0])
    conflicts = list(list(zip(*evaluated_population))[1])

    population = population[:P]
    conflicts = conflicts[:P]

    # Report final results:
    total_population_data = list(zip(population, conflicts))

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

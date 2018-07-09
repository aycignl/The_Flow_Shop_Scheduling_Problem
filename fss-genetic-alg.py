import random
import time
from functools import reduce
import numpy as np


# calculates the total time for execution of all the jobs
# for each job except first, total time passed on the specific machine is:
# max of the time passed until it finishes processing on the previous machine
# or the time passed for the other job to finish on the current machine
# time passed for a job to finish on a machine is:
# the total time passed so far + time it takes to finish task on the machine
def calc_makespan(solution, proccessing_time, number_of_jobs, number_of_machines):
    # list for the time passed until the finishing of the job
    cost = [0] * number_of_jobs
    # for each machine, total time passed will be updated
    for machine_no in range(0, number_of_machines):
        for slot in range(number_of_jobs):
            # time passed so far until the task starts to process
            cost_so_far = cost[slot]
            if slot > 0:
                cost_so_far = max(cost[slot - 1], cost[slot])
            cost[slot] = cost_so_far + proccessing_time[solution[slot]][machine_no]
    return cost[number_of_jobs - 1]

def initialize_population(population_size, number_of_jobs):
    population = []
    i = 0
    while i < population_size:
        individual = list(np.random.permutation(number_of_jobs))
        if individual not in population:
            population.append(individual)
            i += 1

    return population

# Two-point crossover is that the set of jobs between 
# two randomly selected points is always inherited from one parent to the child, 
# and the other jobs are placed in the same manner as the one-point crossover. 
def crossover(parents):
    parent1 = parents[0]
    parent2 = parents[1]
    length_of_parent = len(parent1)
    first_point = int(length_of_parent / 2 - length_of_parent / 4)
    second_point = int(length_of_parent - first_point)
    intersect = parent1[first_point:second_point]

    child = []
    index = 0
    for pos2 in range(len(parent2)):
        if first_point <= index < second_point:
            child.extend(intersect)
            index = second_point
        if parent2[pos2] not in intersect:
                child.append(parent2[pos2])
                index += 1

    return child

# apply mutation to an existing solution using swap move operator
def mutation(solution):
    # copy the solution
    mutated_solution = list(solution)
    solution_length = len(solution)
    # pick 2 positions to swap randomly
    swap_positions = list(np.random.permutation(np.arange(solution_length))[:2])
    first_job = solution[swap_positions[0]]
    second_job = solution[swap_positions[1]]
    mutated_solution[swap_positions[0]] = second_job
    mutated_solution[swap_positions[1]] = first_job
    return mutated_solution

# Selects parent by binary tournament method
def select_parent(population, processing_time, number_of_jobs, number_of_machines):
    parent_pairs = []
    # randomly choose how many parent pairs will be selected
    parent_pair_count = random.randint(2, int(len(population)/2))
    for k in range(parent_pair_count):
        parent1 = binary_tournament(number_of_jobs, number_of_machines, population, processing_time)
        parent2 = binary_tournament(number_of_jobs, number_of_machines, population, processing_time)
        if parent1 != parent2 and (parent1, parent2) not in parent_pairs:
            parent_pairs.append((parent1, parent2))
    return parent_pairs

def binary_tournament(number_of_jobs, number_of_machines, population, processing_time):
    parent = []
    candidates = random.sample(population, 2)
    makespan1 = calc_makespan(candidates[0], processing_time, number_of_jobs, number_of_machines)
    makespan2 = calc_makespan(candidates[1], processing_time, number_of_jobs, number_of_machines)
    if makespan1 < makespan2:
        parent = candidates[0]
    else:
        parent = candidates[1]
    return parent

def update_population(population, children):
    costed_population = []
    for individual in population:
        ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
        costed_population.append(ind_makespan)
    costed_population.sort(key=lambda x: x[0], reverse=True)

    costed_children = []
    for individual in children:
        ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
        costed_children.append(ind_makespan)
    costed_children.sort(key=lambda x: x[0])
    for child in costed_children:
        if child not in population:
            population.append(individual)
            population.remove(costed_population[0][1])
            break

# Start execution

optimal = [4534, 920, 1302]
opt = 0
filename = "data" + str(optimal[opt]) + ".txt"
file = open(filename, 'r')
line = file.readline().split()

# number of jobs and machines
no_of_jobs, no_of_machines = int(line[0]), int(line[1])

# i-th job's processing time at j-th machine 
processing_time = []

for i in range(no_of_jobs):
    temp = []
    line = file.readline().split()
    for j in range(no_of_machines):
        temp.append(int(line[2 * j + 1]))
    processing_time.append(temp)
#print(processing_time)

# generate an initial population proportional to no_of_jobs
number_of_population = no_of_jobs**2
no_of_iterations = 5000
p_crossover = 1.0
p_mutation = 1.0

# Initialize population
population = initialize_population(number_of_population, no_of_jobs)

# Start time for CPU calculation
start_time = time.clock()

for evaluation in range(no_of_iterations):
    # Select parents
    parent_list = select_parent(population, processing_time, no_of_jobs, no_of_machines)
    childs = []

    # Apply crossover to generate children
    for parents in parent_list:
        r = np.random.rand()
        if r < p_crossover:
            childs.append(crossover(parents))
        else:
            if r < 0.5:
                childs.append(parents[0])
            else:
                childs.append(parents[1])

    # Apply mutation operation to change the order of the n-jobs
    mutated_childs = []
    for child in childs:
        r = np.random.rand()
        if r < p_mutation:
            mutated_child = mutation(child)
            mutated_childs.append(mutated_child)

    childs.extend(mutated_childs)
    if len(childs) > 0:
        update_population(population, childs)

# End time for CPU calculation        
end_time = time.clock()

costed_population = []
for individual in population:
    ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
    costed_population.append(ind_makespan)
costed_population.sort(key=lambda x: x[0])

avgObjective = sum(t[0] for t in costed_population) / len(costed_population)
bestObjective = costed_population[0][0]

print("Solution(sequence of jobs processed by the machines):", costed_population[0][1])
print("Objective Value of the Best Chromosome:", str(bestObjective))
print("Average Objective Value of All Chromosomes:", "%.2f" %avgObjective)
print("%Gap of the best chromosome:", "%.2f" % (100 * (bestObjective - optimal[opt]) / optimal[opt]))
print("CPU Time (s)", "%.2f" %(end_time - start_time))

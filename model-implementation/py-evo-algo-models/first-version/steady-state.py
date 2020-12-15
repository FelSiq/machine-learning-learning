"""
	Based on video
	"Vídeo 35 - Computação Evolutiva: Primeiro Modelo"
	URL: https://www.youtube.com/watch?v=JE2S74O0t6Y&t=68s
"""

import matplotlib.pyplot as plt
import collections
import numpy as np
import copy
"""
	Scheduling problem
"""


class Steady:
    """
		Model charactetistics:
		- Each generation: 
			1. a random parent is selected
			2. It always generate a single son via mutation
			3. That son is compared to the instance with less fit of parents population
			4. Repeat n times
		- Mutation: random process is selected, then its core is switched
	"""

    def fitness(self, inst, costs, inverse=False, epsilon=1.0e-16):
        core_workload = [0.0] * len(costs)

        for i in range(len(costs)):
            core_workload[inst[i]] += costs[i]

        max_cost = max(core_workload)
        return 1.0 / (epsilon + max_cost) if inverse else -max_cost

    def mutation(self, inst, cores_num):
        # Swap a random process to another core
        mutated = copy.copy(inst)
        mutated[np.random.randint(len(mutated))] = np.random.randint(cores_num)
        return mutated

    def run(self,
            cores_num,
            costs,
            generation_num=1e+4,
            pop_size=1000,
            random_choice=False,
            ret_stats=False,
            print_stats=True):

        if not isinstance(costs, collections.Iterable):
            raise Exception("Costs parameter should be a real number iterable")

        process_num = len(costs)
        # Generate a random population of size pop_size, each
        # instance being a process_num-length array,
        # allocating a random core for each process
        pop = np.random.randint(cores_num, size=(pop_size, process_num))

        # Calc initial fitness of each instance of
        # the random population
        fitness = np.array([self.fitness(inst, costs) for inst in pop])

        stats = {'fitness': [fitness.mean()], 'deviation': [fitness.std()]}
        gen_id = 0

        while gen_id < generation_num:
            # Select a random parent and apply mutation
            mut_id = np.random.randint(pop_size)
            children = self.mutation(pop[mut_id], cores_num)
            children_fitness = self.fitness(children, costs)

            if not random_choice:
                # Select the less prone instace and check if
                # produced children have higher fitness
                less_prone_id = np.argmin(fitness)
            else:
                # Random choice of a instance to compete
                # with the brand-new children
                less_prone_id = np.random.randint(pop_size)

            smallest_fitness = fitness[less_prone_id]

            if smallest_fitness < children_fitness:
                pop[less_prone_id] = children
                fitness[less_prone_id] = children_fitness

            # Statistics and interface stuff
            avg_fitness = fitness.mean()
            avg_deviation = fitness.std()
            gen_id += 1
            stats['fitness'].append(avg_fitness)
            stats['deviation'].append(avg_deviation)

            if print_stats:
                print(gen_id, ':', avg_fitness, avg_deviation)

        # Return best solution found
        best_fit_id = fitness.argmax()

        if ret_stats:
            return pop[best_fit_id], stats
        return pop[best_fit_id]


"""
	Program driver
"""
if __name__ == '__main__':
    m = Steady()
    costs = [10.0, 50.0, 5, 70.5, 20.0, 20.0, 15, 105, 25]
    sol1, stats1 = m.run(5, costs, ret_stats=True)
    sol2, stats2 = m.run(5, costs, ret_stats=True, random_choice=True)

    print('best solution:', sol1, '(fitness:', m.fitness(sol1, costs), ')')

    plt.subplot(1, 2, 1)
    plt.xlabel("Average Fitness")
    plt.plot(stats1['fitness'])
    plt.plot(stats2['fitness'])
    plt.subplot(1, 2, 2)
    plt.xlabel("Average Deviation")
    plt.plot(stats1['deviation'])
    plt.plot(stats2['deviation'])
    plt.show()

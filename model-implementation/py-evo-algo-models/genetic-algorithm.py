"""
	Based on video
	Vídeo 40 - Computação Evolutiva: Algoritmos Genéticos
	URL: https://www.youtube.com/watch?v=PWqTFbAhAH4
"""

import matplotlib.pyplot as plt
import collections
import numpy as np
import copy

"""
	Scheduling problem
"""

class Genetic:
	"""
		Model charactetistics:
		- Terminology: chromosome = instance
		- Crossover
		- Mutation with probability p: a random process 
			is selected, then its core is switched
	"""

	def fitness(self, inst, costs, inverse=False, epsilon=1.0e-16):
		core_workload=[0.0]*len(costs)

		for i in range(len(costs)):
			core_workload[inst[i]]+=costs[i]

		max_cost=max(core_workload)
		return 1.0/(epsilon+max_cost) if inverse else -max_cost
		

	def mutation(self, inst, cores_num):
		# Swap a random process to another core
		mutated=copy.copy(inst)
		mutated[np.random.randint(len(mutated))]=np.random.randint(cores_num)
		return mutated
	
	def crossover(self, chro_a, chro_b):
		# One point crossover
		n=len(chro_a)//2	
		return np.concatenate((chro_a[:n], chro_b[n:]))

	def run(self, cores_num, costs, generation_num=1e+4, pop_size=1000, 
		mutation_prob=0.05, ret_stats=False, print_stats=True):

		if not isinstance(costs, collections.Iterable):
			raise ValueError("Costs parameter should be a real number iterable")

		if not 0 <= mutation_prob <= 1:
			raise ValueError("Mutation_prob parameter is a probability," +\
				" should be in [0, 1] interval.")

		if len(costs) % 2:
			raise ValueError("This algorithm uses one-point crossover, which needs" +\
				"an cost vector with even length.")

		process_num=len(costs)
		# Generate a random population of size pop_size, each
		# chromosome being a process_num-length array,
		# allocating a random core for each process
		pop=np.random.randint(cores_num, size=(pop_size, process_num))

		# Calc initial fitness of each chromosome of
		# the random population
		fitness=np.array([self.fitness(inst, costs) for inst in pop])

		stats={'fitness': [fitness.mean()], 'deviation': [fitness.std()]}
		gen_id=0

		while gen_id < generation_num:
			chromosomes=[]
			chromosomes_fitness=[]
			for i in range(pop_size):
				# Select two random parents and produce a chromosome
				# with crossover operator
				p_a_id, p_b_id=np.random.randint(pop_size, size=2)
				new_chromosome=self.crossover(pop[p_a_id], pop[p_b_id])
			
				# Each chromosome has a probability of mutation
				if np.random.random() <= mutation_prob:	
					new_chromosome=self.mutation(new_chromosome, cores_num)

				chromosomes.append(new_chromosome)
				chromosomes_fitness.append(self.fitness(new_chromosome, costs))

			# Keep only the new population to the next generation
			pop=np.array(chromosomes)
			fitness=np.array(chromosomes_fitness)

			# Statistics and interface stuff
			avg_fitness=fitness.mean()
			avg_deviation=fitness.std()
			gen_id += 1
			stats['fitness'].append(avg_fitness)
			stats['deviation'].append(avg_deviation)

			if print_stats:
				print(gen_id, ':', avg_fitness, avg_deviation)
		
		# Return best solution found
		best_fit_id=fitness.argmax()

		if ret_stats:
			return pop[best_fit_id], stats
		return pop[best_fit_id]	

"""
	Program driver
"""
if __name__ == '__main__':
	m=Genetic()
	#costs=[10.0, 50.0, 5, 70.5, 20.0, 20.0, 15, 105, 25]
	costs=np.random.random(100) * 400 + 100
	sol, stats=m.run(25, costs, generation_num=250, pop_size=1000, ret_stats=True)

	print('best solution:', sol, '(fitness:', m.fitness(sol, costs), ')')

	plt.subplot(1, 2, 1)
	plt.xlabel("Average Fitness")
	plt.plot(stats['fitness'])
	plt.subplot(1, 2, 2)	
	plt.xlabel("Average Deviation")
	plt.plot(stats['deviation'])
	plt.show()

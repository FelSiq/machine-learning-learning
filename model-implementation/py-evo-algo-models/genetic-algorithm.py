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
		Model characteristics:
		- Choice two parents with a probability proportional of its fitness
		- Produce two children with Crossover operator
		- Keep only childrens to the next generation
		- Each children has a probability of mutation of its own

		- Terminology: chromosome = instance
		- Mutation with probability p: a random process 
			is selected, then its core is switched
	"""

	def fitness(self, inst, costs, epsilon=1.0e-16):
		core_workload=[0.0]*len(costs)

		for i in range(len(costs)):
			core_workload[inst[i]]+=costs[i]

		max_cost=max(core_workload)
		return 1.0/(epsilon+max_cost)
		

	def mutation(self, inst, cores_num):
		# Swap a random process to another core
		mutated=copy.copy(inst)
		mutated[np.random.randint(len(mutated))]=np.random.randint(cores_num)
		return mutated
	
	def crossover(self, chro_a, chro_b):
		# One point crossover
		# Supposing chro_a and chro_b has equals even lengths
		cut_point=len(chro_a)//2	
		new_chro_a=np.concatenate((chro_a[:cut_point], chro_b[cut_point:]))
		new_chro_b=np.concatenate((chro_b[:cut_point], chro_a[cut_point:]))
		return new_chro_a, new_chro_b
		

	def run(self, cores_num, costs, generation_num=1e+4, pop_size=1000, 
		mutation_prob=0.05, weighted_choice=True, ret_stats=False, print_stats=True):

		if not isinstance(costs, collections.Iterable):
			raise ValueError("Costs parameter should be a real number iterable")

		if not 0 <= mutation_prob <= 1:
			raise ValueError("Mutation_prob parameter is a probability," +\
				" should be in [0, 1] interval.")

		if len(costs) == 0 or len(costs) % 2:
			raise ValueError("This algorithm uses one-point crossover, which needs" +\
				"an cost vector with positive even length.")

		if pop_size <= 0 or pop_size % 2:
			raise ValueError("This algorithm uses sexual reproduction, which needs" +\
				"a pop_size parameter with an positive even value.")

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
		
			# Probability of each parent be selected to reproduce
			if weighted_choice:
				prob_vector=fitness/sum(fitness)
			else:
				prob_vector=[1.0/pop_size] * pop_size

			for i in range(pop_size//2):
				# Select two random parents and produce a chromosome
				# with crossover operator. The probability to choose
				# each chromosome is proportional of its fitness.
				p_a_id, p_b_id=np.random.choice(pop_size, size=2, p=prob_vector)
				new_chro_a, new_chro_b=self.crossover(pop[p_a_id], pop[p_b_id])
			
				# Each chromosome has a probability of mutation
				if np.random.random() <= mutation_prob:	
					new_chro_a=self.mutation(new_chro_a, cores_num)
				if np.random.random() <= mutation_prob:	
					new_chro_b=self.mutation(new_chro_b, cores_num)

				chromosomes.append(new_chro_a)
				chromosomes_fitness.append(self.fitness(new_chro_a, costs))

				chromosomes.append(new_chro_b)
				chromosomes_fitness.append(self.fitness(new_chro_b, costs))

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
	costs=np.random.random(100) * 400 + 100
	sol1, stats1=m.run(25, costs, generation_num=200, pop_size=1000, ret_stats=True)
	sol2, stats2=m.run(25, costs, generation_num=200, pop_size=1000, ret_stats=True, weighted_choice=False)

	print("best solution:", sol1, "(fitness:", m.fitness(sol1, costs), ")")

	plt.suptitle("Genetic algorithm")
	plt.subplot(1, 2, 1)
	plt.xlabel("Average Fitness")
	plt.plot(stats1["fitness"], label="Weighted-prob avg fitness")
	plt.plot(stats2["fitness"], label="Random-choice avg fitness")
	plt.legend(loc="upper left")
	plt.subplot(1, 2, 2)	
	plt.xlabel("Average Deviation")
	plt.plot(stats1["deviation"], label="Weighted-prob avg deviation")
	plt.plot(stats2["deviation"], label="Random-choice avg deviation")
	plt.legend(loc="lower left")
	plt.show()

"""
	Based on video
	"Vídeo 37 - Computação Evolutiva: Modelo em Batch ou de Geração"
	URL: https://www.youtube.com/watch?v=Qn1mPCoz5yQ
"""

import matplotlib.pyplot as plt
import collections
import numpy as np
import copy

"""
	Scheduling problem
"""

class Batch:
	"""
		Model charactetistics:
		- For each generation
			-> choice K random parents
			-> Each parent produce a children via mutation
			-> Keep each children in a separated population
			-> for each produced children, select a random instance
				from its parent population
			-> keep the fittest instance
		- Repeat

		- Mutation: random process is selected, then its core is switched
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

	def run(self, cores_num, costs, sons_per_gen=10, generation_num=1e+4, pop_size=1000, 
		random_choice=False, ret_stats=False, print_stats=True):

		if not isinstance(costs, collections.Iterable):
			raise Exception("Costs parameter should be a real number iterable")

		process_num=len(costs)
		# Generate a random population of size pop_size, each
		# instance being a process_num-length array,
		# allocating a random core for each process
		pop=np.random.randint(cores_num, size=(pop_size, process_num))

		# Calc initial fitness of each instance of
		# the random population
		fitness=np.array([self.fitness(inst, costs) for inst in pop])

		stats={'fitness': [fitness.mean()], 'deviation': [fitness.std()]}
		gen_id=0

		while gen_id < generation_num:
			childrens=[]
			childrens_fitness=[]
			for i in range(sons_per_gen):
				# Select a random parent and apply mutation
				mut_id=np.random.randint(pop_size)
				new_children=self.mutation(pop[mut_id], cores_num)
				childrens.append(new_children)
				childrens_fitness.append(self.fitness(new_children, costs))

			for i in range(sons_per_gen):
				if not random_choice:
					# Select the less prone instance and check if
					# produced children have higher fitness
					less_prone_id=np.argmin(fitness)
				else:
					# Random choice of a instance to compete
					# with the brand-new children
					less_prone_id=np.random.randint(pop_size)

				smallest_fitness=fitness[less_prone_id]

				if smallest_fitness < childrens_fitness[i]:
					pop[less_prone_id]=childrens[i]
					fitness[less_prone_id]=childrens_fitness[i]

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
	m=Batch()
	costs=np.random.random(100) * 400 + 100
	sol1, stats1=m.run(25, costs, generation_num=1000, pop_size=10000, ret_stats=True)
	sol2, stats2=m.run(25, costs, generation_num=1000, pop_size=10000, ret_stats=True, random_choice=True)

	print('best solution:', sol1, '(fitness:', m.fitness(sol1, costs), ')')

	plt.suptitle("Batch model")
	plt.subplot(1, 2, 1)
	plt.xlabel("Average Fitness")
	plt.plot(stats1["fitness"], label="Fittest-choice avg fitness")
	plt.plot(stats2["fitness"], label="Random-choice avg fitness")
	plt.legend(loc="upper left")
	plt.subplot(1, 2, 2)	
	plt.xlabel("Average Deviation")
	plt.plot(stats1["deviation"], label="Fittest-choice avg deviation")
	plt.plot(stats2["deviation"], label="Random-choice avg deviation")
	plt.legend(loc="upper right")
	plt.show()


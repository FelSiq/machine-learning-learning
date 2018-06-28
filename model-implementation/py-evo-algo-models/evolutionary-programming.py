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

class EvolProg:
	"""
		Model charactetistics:
		-
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

	def run(self, cores_num, costs, generation_num=1e+4, pop_size=1000, 
		ret_stats=False, print_stats=True):

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
			for i in range(pop_size):
				# Select a each parent and apply mutation
				new_children=self.mutation(pop[i], cores_num)
				childrens.append(new_children)
				childrens_fitness.append(self.fitness(new_children, costs))

			# Unify both populations and keep only the
			# top pop_size best instances (with higher fitness)
			fitness=np.concatenate((fitness, childrens_fitness))
			pop=np.concatenate((pop, childrens))

			fitness, pop=map(np.array, zip(*sorted(zip(fitness, pop), \
				key=lambda key : key[0], reverse=True)[:pop_size]))

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
	m=EvolProg()
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

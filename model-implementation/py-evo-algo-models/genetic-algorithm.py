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
		- Each children has a probability of mutation of its own
		- Keep only childrens to the next generation
		- If elitism is applied, then the fittest parent will replace a single
			random children of the new population

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
		mutation_prob=0.05,  # Probability of each new children to suffer mutation
		weighted_choice=True, # Should parents be selected with prob proportional of its fitness?
		elitism=1, # Quantity of the top fittest parents to be kept to the next generation
		ranking=False, # Should the fitness of each population be replaced to a ranking-based score? 
		tournament=False, # Should parents be selected via a tournament?
		tournament_size=5, # If yes, whats the tournament size?
		ret_stats=True,
		print_stats=True):

		if not isinstance(costs, collections.Iterable):
			raise ValueError("Costs parameter should be a real number iterable")

		if not 0 <= mutation_prob <= 1:
			raise ValueError("Mutation_prob parameter is a probability," +\
				" should be in [0, 1] interval.")

		if len(costs) == 0 or len(costs) % 2:
			raise ValueError("This algorithm uses one-point crossover, which needs" +\
				" an cost vector with positive even length.")

		if pop_size <= 0 or pop_size % 2:
			raise ValueError("This algorithm uses sexual reproduction, which needs" +\
				" a pop_size parameter with an positive even value.")

		if tournament and weighted_choice:
			raise ValueError("Tournament and weighted_choice parameters" +\
				" are mutually exclusive.")

		if tournament and ranking:
			print("Warning: ranking will be ignored in tournament method.")

		process_num=len(costs)
		# Generate a random population of size pop_size, each
		# chromosome being a process_num-length array,
		# allocating a random core for each process
		pop=np.random.randint(cores_num, size=(pop_size, process_num))

		# Calc initial fitness of each chromosome of
		# the random population
		fitness=np.array([self.fitness(inst, costs) for inst in pop])

		stats={"fitness": [fitness.mean()], "deviation": [fitness.std()]}
		gen_id=0

		while gen_id < generation_num:

			# Ranking-based stuff is not tournament compatible
			if not tournament:
				if weighted_choice:
					# Should the fitness be replaced to a ranking-based score?
					# This helps dominance of a few chromosomes alongside the
					# generations, as it imply in reducing the search on the
					# solution space.
					if ranking:
						# First, sort population by its fitness
						pop, fitness = zip(*sorted(zip(pop, fitness), \
							key= lambda key : key[1]))

						# Then replace each fitness with a score based
						# on its current postion. Higher positions should
						# have the higher ranking score.
						total=0.5 * pop_size * (1 + pop_size) # Arithmetic progression sum
						fitness=np.array([i for i in range(1, pop_size+1)])/total
				
					# Probability of each parent be selected to reproduce
					prob_vector=fitness/sum(fitness)
				else:
					# Probability of each parent be selected to reproduce
					prob_vector=[1.0/pop_size] * pop_size

			chromosomes=[]
			chromosomes_fitness=[]
		
			for i in range(pop_size//2):
				if not tournament:
					# Select two random parents and produce a chromosome
					# with crossover operator. The probability to choose
					# each chromosome is proportional of its fitness. If
					# Ranking method is used, this proportionallity will
					# be linear for between chromosomes.
					p_a_id, p_b_id=np.random.choice(pop_size, size=2, \
						p=prob_vector, replace=False)
				else:
					# The tournament with size k works as follows:
					# Select k random candidates of the population to the tournament
					# Get the two fittest ones
					# They are the selected ones to produce a children with crossover
					candidates_id=np.random.choice(pop_size, replace=False, \
						size=tournament_size)

					candidates_id, candidate_fitness=zip(*sorted(zip(\
						candidates_id, fitness[candidates_id]),\
						key=lambda key : key[1], reverse=True))

					p_a_id, p_b_id=candidates_id[:2]

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

			# If elitism is applied, keep the best parent chromosome
			# to the next generation
			for i in range(elitism):
				fittest_parent=np.argmax(fitness)
				children_id=np.random.randint(pop_size)
				chromosomes_fitness[children_id]=fitness[fittest_parent]
				chromosomes[children_id]=pop[fittest_parent]
				fitness[fittest_parent]=-np.inf

			# Keep only the new population to the next generation
			# except for the fittest ones, if elitism method is applied
			pop=np.array(chromosomes)
			fitness=np.array(chromosomes_fitness)

			# Statistics and interface stuff
			avg_fitness=fitness.mean()
			avg_deviation=fitness.std()
			gen_id += 1
			stats["fitness"].append(avg_fitness)
			stats["deviation"].append(avg_deviation)

			if print_stats:
				print(gen_id, ":", avg_fitness, avg_deviation)
		
		# Return best solution found
		best_fit_id=fitness.argmax()

		if ret_stats:
			return pop[best_fit_id], stats
		return pop[best_fit_id]	

"""
	Program driver
"""
if __name__ == "__main__":
	m=Genetic()
	costs=np.random.random(100) * 400 + 100
	gen_num=400

	sol1, stats1=m.run(25, costs, generation_num=gen_num, pop_size=1000, 
		ret_stats=True, elitism=False, ranking=True)
	#sol2, stats2=m.run(25, costs, generation_num=200, pop_size=1000, 
	#	ret_stats=True, elitism=False, ranking=False)
	sol3, stats3=m.run(25, costs, generation_num=gen_num, pop_size=1000, 
		ret_stats=True, elitism=True, ranking=True)
	#sol4, stats4=m.run(25, costs, generation_num=200, pop_size=1000, 
	#	ret_stats=True, elitism=True, ranking=False)
	sol5, stats5=m.run(25, costs, generation_num=gen_num, pop_size=1000, 
		ret_stats=True, elitism=False, tournament=True, weighted_choice=False)
	sol6, stats6=m.run(25, costs, generation_num=gen_num, pop_size=1000, 
		ret_stats=True, elitism=True, tournament=True, weighted_choice=False)

	plt.suptitle("Genetic algorithm")
	plt.subplot(1, 2, 1)
	plt.xlabel("Average Fitness")
	plt.plot(stats1["fitness"], label="With ranking")
	#plt.plot(stats2["fitness"], label="Rankingless")
	plt.plot(stats3["fitness"], label="With ranking+Elitism")
	#plt.plot(stats4["fitness"], label="Rankingless+Elitism")
	plt.plot(stats5["fitness"], label="Tournament")
	plt.plot(stats6["fitness"], label="Tournament+Elitism")
	plt.legend(loc="upper left")
	plt.subplot(1, 2, 2)	
	plt.xlabel("Average Deviation")
	plt.plot(stats1["deviation"], label="With ranking")
	#plt.plot(stats2["deviation"], label="Rankingless")
	plt.plot(stats3["deviation"], label="With ranking+Elitism")
	#plt.plot(stats4["deviation"], label="Rankingless+Elitism")
	plt.plot(stats5["deviation"], label="Tournament")
	plt.plot(stats6["deviation"], label="Tournament+Elitism")
	plt.legend(loc="upper right")
	plt.show()

	"""
	sol2, stats2=m.run(25, costs, generation_num=200, pop_size=1000, 
		ret_stats=True, weighted_choice=False, elitism=False)
	
	sol3, stats3=m.run(25, costs, generation_num=200, pop_size=1000, 
		ret_stats=True, elitism=True)

	print("best solution:", sol1, "(fitness:", m.fitness(sol1, costs), ")")

	plt.suptitle("Genetic algorithm")
	plt.subplot(1, 2, 1)
	plt.xlabel("Average Fitness")
	plt.plot(stats1["fitness"], label="Weighted-prob")
	plt.plot(stats2["fitness"], label="Random-choice")
	plt.plot(stats3["fitness"], label="Weighted+Elitism")
	plt.legend(loc="upper left")
	plt.subplot(1, 2, 2)	
	plt.xlabel("Average Deviation")
	plt.plot(stats1["deviation"], label="Weighted-prob")
	plt.plot(stats2["deviation"], label="Random-choice")
	plt.plot(stats3["deviation"], label="Weighted+Elitism")
	plt.legend(loc="lower left")
	plt.show()
	"""

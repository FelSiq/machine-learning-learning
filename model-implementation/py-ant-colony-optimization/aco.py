import matplotlib.pyplot as plt
import numpy as np
import sys

class Aco:
	def run(self, 
		graph, 
		ants_num=100, 
		start_id=0, 
		end_id=-1,
		num_generations=250, 
		pheromone_inc=1.0,
		pheromone_fade_ratio=0.1, 
		epsilon=1.0e-16,
		iterative_plot=False):

		"""
			This algorithm assumes that higher edge
			values are better ways than lower ones.
		"""

		nrow, ncol=graph.shape

		if nrow != ncol:
			raise ValueError("Graph must be a square matrix.")

		if not 0 <= pheromone_fade_ratio <= 1.0:
			raise ValueError("Pheromone_fade_ratio must be in [0, 1].")

		if end_id < 0:
			end_id = ncol-1

		pheromone_trace=np.array([[epsilon] * ncol \
			for _ in range(nrow)])

		ant_position=[0] * ants_num
		ant_onward=[True] * ants_num

		gen_id=0
		while gen_id < num_generations:
			# Clean up pheromone_update axiliary matrix
			pheromone_update=np.array([[0.0] * ncol \
				for _ in range(nrow)])

			# Move ants based on the pheromones
			for ant_id in range(ants_num):
				cur_ant_pos=ant_position[ant_id]

				if ant_onward[ant_id]:
					# Assume that graph higher positions (relative to 
					# the current position) are always a onward position
					next_dir_probs=graph[cur_ant_pos, (cur_ant_pos+1):]
					next_dir_probs/=sum(next_dir_probs)
					next_position=np.random.choice(\
						range(cur_ant_pos+1, ncol),
						p=next_dir_probs)
				else:
					# Assume that graph lower positions (relative to 
					# the current position) are always a backward position
					next_dir_probs=graph[cur_ant_pos, :cur_ant_pos]
					next_dir_probs/=sum(next_dir_probs)
					next_position=np.random.choice(\
						range(cur_ant_pos),
						p=next_dir_probs)

				# Update next position and pheromone ratios
				ant_position[ant_id]=next_position
				pheromone_update[cur_ant_pos, next_position]+=pheromone_inc
				pheromone_update[next_position, cur_ant_pos]+=pheromone_inc

				if next_position == end_id or next_position == start_id:
					# If ant reached it's destiny or is in the nest, 
					# change it's direction
					ant_onward[ant_id] = not ant_onward[ant_id]


			# Update pheromone matrix
			pheromone_trace = (1.0 - pheromone_fade_ratio) * \
				pheromone_trace + pheromone_update

			gen_id += 1
			print("\rGen id:", gen_id, "...", end="")

		print("\nCompleted.")
		return pheromone_trace

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("usage:", sys.argv[0], "<graph filepath>")
		exit(1)

	model=Aco()
	graph=np.loadtxt(sys.argv[1], delimiter=" ")
	print(graph)
	pheromone_trace=model.run(graph, iterative_plot=True)

	print("Pheromone matrix:\n", pheromone_trace)

	plt.imshow(pheromone_trace, cmap="hot") 
	plt.title("Pheromone trace heatmap")
	plt.show()

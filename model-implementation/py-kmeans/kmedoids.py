from numpy import array, random, argmin, mean as npmean, inf
from pandas import DataFrame, read_csv
from clustering.metrics import ClusterMetrics

"""
	A "Medoid", different from a "Centroid", MUST BE a
	instance from the dataset.
"""

class Kmedoids():
	def __euclideandist__(inst_a, inst_b):
		return (sum((inst_a - inst_b)**2.0))**0.5

	def run(dataset, k, it_max=1000, min_variation=1.0e-4, labels=None):
		# Init medoids at random
		medoids_ids = random.randint(dataset.shape[0], size=k)
		medoids_coord = dataset[medoids_ids,:]

		# Each instance will be put on a initial random cluster
		inst_cluster_id = random.randint(k, size=dataset.shape[0])
		
		# Auxiliary vectors to keep code cleaner
		auxvec_cur_distances = array([0.0] * k)

		prev_variation = 1.0 + min_variation
		it = 0
		while it < it_max and prev_variation >= min_variation:
			it += 1

			for inst_id in range(dataset.shape[0]):
				nearest_medoid = inst_cluster_id[inst_id]
				for medoid_id in range(k):
					# For each instance, calculate the distance between
					# each medoid
					auxvec_cur_distances[medoid_id] = \
						Kmedoids.__euclideandist__(\
							dataset[inst_id,:],
							medoids_coord[medoid_id,:])
					
				# For each instance, let it be part of the nearest
				# cluster
				inst_cluster_id[inst_id] = argmin(auxvec_cur_distances)

			# For each cluster, calculate the new medoid coordinates
			for medoid_id in range(k):

				# Calc the cumulative sum of the distances of each
				# instance to all instances within the same cluster
				min_cumsum = inf
				next_medoid_id = medoid_id
				next_medoid_coords = medoids_ids[medoid_id]

				for inst_id in range(dataset.shape[0]):
					if inst_cluster_id[inst_id] == medoid_id:
						cur_cumsum = sum((((dataset[inst_cluster_id == medoid_id,:] - \
								dataset[inst_id,:])**2.0).sum(axis=1))**0.5)

						# The next medoid of the current cluster which
						# minimizes the cumulative distance between all
						# other instances
						if cur_cumsum < min_cumsum:
							min_cumsum = cur_cumsum
							next_medoid_id = inst_id
							next_medoid_coords = dataset[inst_id,:]

				# Calculate variation between previous medoids_coord and
				# new ones (using infinite norm)
				prev_variation = max(prev_variation, \
					max(abs(medoids_coord[medoid_id] - \
						next_medoid_coords)))

				medoids_coord[medoid_id] = next_medoid_coords
				medoids_ids[medoid_id] = next_medoid_id

		# Build up answer
		ans = {
			"medoids_coord" : medoids_coord,
			"medoids_ids" :	medoids_ids,
			"clusters" : inst_cluster_id,
			"inner_metrics" : {
				"SSE/Cohesion" : ClusterMetrics.sse(dataset, \
					medoids_coord, inst_cluster_id),
				"BSS/Separation" : ClusterMetrics.bss(dataset, \
					medoids_coord, inst_cluster_id),
				"Silhouette" : ClusterMetrics.silhouette(dataset, \
					medoids_coord, inst_cluster_id),
			},
		}

		if labels is not None:
			# If true labels are given, then we can compute
			# OUTTER clustering quality measures
			ans = {
				**ans,
				"outter_metrics" : {
					"Rand_index" : ClusterMetrics.rand_index(\
						dataset, labels, inst_cluster_id),
					"Adjusted_rand_index" : ClusterMetrics.adjusted_rand_index(\
						dataset, labels, inst_cluster_id),
					"Jackard_index" : ClusterMetrics.jackard_index(\
						dataset, labels, inst_cluster_id),
				}
			}

		return ans

if __name__ == "__main__":
	import sys

	if len(sys.argv) < 3:
		print("usage:", sys.argv[0], "<data_filepath> <k>",
			"\n\t[-sep data_separator] [-label column_label_to_remove]")
		exit(1)

	try:
		sep = sys.argv[1 + sys.argv.index("-sep")]
	except:
		sep = ","

	dataset = read_csv(sys.argv[1], sep=sep)

	try:
		rem_label = sys.argv[1 + sys.argv.index("-label")]
		class_ids = dataset.pop(rem_label)
	except:
		class_ids = None
		if ("-label",) in sys.argv:
			print("Warning: can not remove column \"" +\
				rem_label + "\" from dataset.")
	ans = Kmedoids.run(
		dataset=dataset.loc[:,:].values, 
		k=int(sys.argv[2]),
		labels=class_ids)

	print("Results:")
	for item in ans:
		if type(ans[item]) == type({}):
			print(item, ": ", sep="")
			for val in ans[item]:
				print("\t", val, ": ", 
					ans[item][val], sep="")
		else:
			print(item, ":\n", ans[item], sep="")
		print()



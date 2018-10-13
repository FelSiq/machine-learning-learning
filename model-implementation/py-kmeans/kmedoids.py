from numpy import array, random, argmin, mean as npmean, inf

import sys
sys.path.insert(0, "../../validation-framework/")
from clustering import ClusterMetrics

"""
	A "Medoid", different from a "Centroid", MUST BE a
	instance from the dataset.
"""

class Kmedoids():
	def __euclideandist__(inst_a, inst_b):
		return (sum((inst_a - inst_b)**2.0))**0.5

	def run(dataset, k,
		it_max=1000, 
		min_variation=1.0e-4, 
		labels=None, 
		full_output=True):

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
			"inst_cluster_id" : inst_cluster_id, 
			"centers_coord" : medoids_coord,
		}

		if full_output:
			ans = {
				"clustering_method" : "K-Medoids",
				"k" : k,
				**ans,
				**ClusterMetrics.runall(\
					dataset=dataset,
					centers_coord=medoids_coord,
					inst_cluster_id=inst_cluster_id,
					labels=labels),
			}

		return ans

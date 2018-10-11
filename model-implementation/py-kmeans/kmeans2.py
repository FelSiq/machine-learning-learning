# Helpful video: (in Portuguese)
# title: "Vídeo 48 - Planejamento de Experimentos: Agrupamento de Dados"
# canal: ML4U
# https://www.youtube.com/watch?v=OLwabj1WJj0

from numpy import array, random, argmin, mean as npmean, inf
from pandas import DataFrame, read_csv
from clustering.metrics import ClusterMetrics

class Kmeans():
	def __euclideandist__(inst_a, inst_b):
		return (sum((inst_a - inst_b)**2.0))**0.5

	def run(dataset, k, it_max=1000, min_variation=1.0e-4, labels=None):
		# Init centers_coord
		centers_id = random.randint(dataset.shape[0], size=k)
		centers_coord = dataset[centers_id,:]

		# Each instance will be put on a initial random cluster
		inst_cluster_id = random.randint(k, size=dataset.shape[0])
		
		# Auxiliary vectors to keep code cleaner
		auxvec_cur_distances = array([0.0] * k)

		prev_variation = 1.0 + min_variation
		it = 0
		while it < it_max and prev_variation >= min_variation:
			it += 1

			for inst_id in range(dataset.shape[0]):
				nearest_center = inst_cluster_id[inst_id]
				for center_id in range(k):
					# For each instance, calculate the distance between
					# each center
					auxvec_cur_distances[center_id] = \
						Kmeans.__euclideandist__(\
							dataset[inst_id,:],
							centers_coord[center_id,:])
					
				# For each instance, let it be part of the nearest
				# cluster
				inst_cluster_id[inst_id] = argmin(auxvec_cur_distances)

			# For each cluster, calculate the new center coordinates
			for center_id in range(k):
				new_cur_cluster_coords = npmean(dataset[\
					inst_cluster_id == center_id,:], axis=0)

				# Calculate variation between previous centers_coord and
				# new ones (using infinite norm)
				prev_variation = max(prev_variation, \
					max(abs(centers_coord[center_id] - \
						new_cur_cluster_coords)))

				centers_coord[center_id] = new_cur_cluster_coords

		# Build up answer
		ans = {
			"centers" : centers_coord,
			"clusters" : inst_cluster_id,
			"inner_metrics" : {
				"SSE/Cohesion" : ClusterMetrics.sse(dataset, \
					centers_coord, inst_cluster_id),
				"BSS/Separation" : ClusterMetrics.bss(dataset, \
					centers_coord, inst_cluster_id),
				"Silhouette" : ClusterMetrics.silhouette(dataset, \
					centers_coord, inst_cluster_id),
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
	ans = Kmeans.run(
		dataset=dataset.loc[:,:].values, 
		k=int(sys.argv[2]),
		labels=class_ids)

	print("Results:")
	for item in ans:
		if type(ans[item]) == type({}):
			print(item, ":", sep="")
			for val in ans[item]:
				print("\t", val, ":", 
					ans[item][val], sep="")
		else:
			print(item, ":\n", ans[item], sep="")
		print()



# Helpful video: (in Portuguese)
# title: "Vídeo 48 - Planejamento de Experimentos: Agrupamento de Dados"
# canal: ML4U
# https://www.youtube.com/watch?v=OLwabj1WJj0

from numpy import array, random, argmin, mean as npmean, where, inf
from pandas import DataFrame, read_csv

class Kmeans():
	def __euclideandist__(inst_a, inst_b):
		return (sum((inst_a - inst_b)**2.0))**0.5

	def sse(dataset, centers_coord, inst_cluster_id):
		# SEE : Sum of Squared Errors
		# It measures Cluster's Cohesion quality
		# It is a INNER CRITERION for unsupervised quality metrics

		center_num = len(centers_coord)

		sse_value = 0.0
		for center_id in range(center_num):
			this_cluster_insts = where(inst_cluster_id == center_id)[0]
			cluster_mean = npmean(dataset[this_cluster_insts,:], axis=0)
			sse_value += max(sum((dataset[this_cluster_insts,:] \
				- cluster_mean)**2.0))

		return sse_value

	def bss(dataset, centers_coord, inst_cluster_id):
		# BSS : Between Sum of Squares
		# It measures Separation between clusters
		# It is a INNER CRITERION for unsupervised quality metrics

		center_num = len(centers_coord)

		dataset_mean = dataset.mean(axis=0)

		bss_value = 0.0
		for center_id in range(center_num):
			this_cluster_insts = where(inst_cluster_id == center_id)[0]
			cluster_mean = npmean(dataset[this_cluster_insts,:], axis=0)
			sqr_coord_diffs = (dataset_mean - cluster_mean)**2.0
			bss_value += len(this_cluster_insts) * max(sqr_coord_diffs)

		return bss_value

	def silhouette(dataset, centers_coord, inst_cluster_id):
		# Silhouette is a way of summarize the BSS and SSE
		# metrics into a single measure value. So, obvious enough,
		# it is also a INNER CRITERION for unsupervised quality metrics.
		inst_num = dataset.shape[0]
		inst_sil = array([0.0] * inst_num)
		center_num = len(centers_coord)

		for center_id in range(center_num):
			this_cluster_insts = \
				where(inst_cluster_id == center_id)[0]

			for inst_id in this_cluster_insts:
				# Calculate average distance of this 
				# instance with every other instance
				# of the SAME cluster
				inst_avg_dist_intracluster = \
					max(abs(npmean(dataset[this_cluster_insts,:] -\
						dataset[inst_id,:], axis=0)))

				# Calculate the MINIMAL average distance
				# of this instance with every other ins-
				# tance of DIFFERENT clusters for each 
				# other cluster
				inst_min_avg_dist_intercluster = inf
				for i in range(center_num):
					if i != center_id:
						inst_min_avg_dist_intercluster =\
							min(inst_min_avg_dist_intercluster, \
								max(abs(npmean(dataset[inst_cluster_id == i,:] -\
									dataset[inst_id,:], axis=0))))

				# Calculate this instance silhouette 
				inst_sil[inst_id] = \
					(inst_min_avg_dist_intercluster - \
						inst_avg_dist_intracluster)/\
					max(inst_min_avg_dist_intercluster, \
						inst_avg_dist_intracluster)

		return npmean(inst_sil)

	def run(dataset, k, it_max=1000, min_variation=1.0e-4):
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
				new_cur_cluster_coords = npmean(dataset[inst_cluster_id == center_id,:], axis=0)

				# Calculate variation between previous centers_coord and
				# new ones (using infinite norm)
				prev_variation = max(prev_variation, \
					max(abs(centers_coord[center_id] - new_cur_cluster_coords)))

				centers_coord[center_id] = new_cur_cluster_coords

		# Build up answer
		ans = {
			"centers" : centers_coord,
			"clusters" : inst_cluster_id,
			"inner_metrics" : {
				"SSE/Cohesion" : Kmeans.sse(dataset, centers_coord, inst_cluster_id),
				"BSS/Separation" : Kmeans.bss(dataset, centers_coord, inst_cluster_id),
				"Silhouette" : Kmeans.silhouette(dataset, centers_coord, inst_cluster_id),
			},
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
		k=int(sys.argv[2]))

	if class_ids is not None:
		# Given class labels, one can calculate
		# OUTTER unsupervised measures like
		# - Rand Index
		# - Jackard Index
		# - Adjusted Rand Index
		# Will be coded eventually.
		pass

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



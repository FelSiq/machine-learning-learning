from numpy import where, mean as npmean, array, inf
from scipy.special import binom

class ClusterMetrics():
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

	def rand_index(
		labels, 
		inst_cluster_id, 
		return_counters=False): 

		if len(labels) != len(inst_cluster_id):
			if warnings:
				print("Error: length of label and",
					"inst_clust_id must match.")
			return None

		"""
			Rand Index definition:
			let
			f_00 : # of instance pairs with diff classes in diff clusters
			f_01 : # of instance pairs with diff classes in the same cluster
			f_10 : # of instance pairs with same class in diff clusters
			f_11 : # of instance pairs with same classes in the same cluster

			Therefore, the most significative "bit" tells about 
			the class label pair and the less significative one
			about the cluster pair.

					Same cluster	Diff Cluster
			Same class	f_11		f_10
			Diff class	f_01		f_00

			then:
			rand_index := (f_00 + f_11) / (f_00 + f_10 + f_01 + f_11)
		"""

		# The counter array will express the f_ij value
		# in terms of it's own index. (For example, the 
		# counter f_10 is counters[int("10", 2)] == counters[2]
		# and the counter f_11 is counters[int("11", 2)] == counters[3])
		counters = 4 * [0]

		num_inst = len(labels)
		for i in range(num_inst):
			for j in range(num_inst):
				pos = 2 * (labels[i] == labels[j]) + \
					(inst_cluster_id[i] == inst_cluster_id[j])
				counters[pos] += 1

			# Remove the i == j counter. I do not put an
			# extra "if" to speed up the process.
			counters[3] -= 1 # Remember: int("11", 2) == 3.

		# rand_index = (f_00 + f_11) / (f_00 + f_10 + f_01 + f_11)
		rand_index = (counters[int("00", 2)] + \
			counters[int("11", 2)]) / sum(counters)

		if return_counters:
			return rand_index, counters

		return rand_index
		
	
	def adjusted_rand_index(labels, inst_cluster_id): 

		if len(set(labels)) != len(set(inst_cluster_id)):
			return None

		"""
			The first problem that arrives calculting this
			measure is the relation between the clusters ID and
			the true instance labels. For example, picking
			up some Iris dataset instances:

			11,5.4,3.7,1.5,0.2,setosa
			50,5.0,3.3,1.4,0.2,setosa
			51,7.0,3.2,4.7,1.4,versicolor
			86,6.0,3.4,4.5,1.6,versicolor
			113,6.8,3.0,5.5,2.1,virginica
			121,6.9,3.2,5.7,2.3,virginica

			Supposing a 100% accurary from he Kmeans algorithm,
			can map these instances with any combination of 0's,
			1's and 2's given their classes:

			Possibility 1: [0 0 1 1 2 2]
			Possibility 2: [1 1 0 0 2 2]
			Possibility 3: [0 0 2 2 1 1]
			Possibility 4: [1 1 2 2 0 0]
			Possibility 5: [2 2 0 0 1 1]
			Possibility 6: [2 2 1 1 0 0]

			This means that the first step is top discover which
			relationship the class labels and the cluster IDs ha-
			ve. One way of doing this is to check which cluster ID
			one class has more connected instances with.
		"""
		
		# Discover the relationship between the cluster IDs and
		# the class labels
		class_label_set = set(labels)
		clusters_id_set = set(inst_cluster_id)

		class_cluster_mapping = {}
		aux_clusters_id_set = list(clusters_id_set)

		for class_label in class_label_set:
			remaining_clusters = len(aux_clusters_id_set)
			intersec_cardinality_list = array([0] * remaining_clusters)
			for cluster_id in range(remaining_clusters):
				intersec_cardinality_list[cluster_id] = sum(\
					(inst_cluster_id == aux_clusters_id_set[cluster_id]) &\
					(labels == class_label))

			best_choice_id = aux_clusters_id_set[\
				intersec_cardinality_list.argmax()]
			class_cluster_mapping[class_label] = best_choice_id
			aux_clusters_id_set.remove(best_choice_id)

		# Build up the contingency table
		class_fixed_seq = class_cluster_mapping.keys()
		cluster_fixed_seq = [class_cluster_mapping[key] \
			for key in class_fixed_seq]

		contingency_table = array([[
			sum((labels == cla_label) & \
				(inst_cluster_id == clu_id))
			for cla_label in class_fixed_seq]
			for clu_id in cluster_fixed_seq])

		# Calculate the row and column sums
		b_col_sum = contingency_table.sum(axis=0)
		a_row_sum = contingency_table.sum(axis=1)

		# Calculate the Adjusted Rand Index
		N = len(labels)
		aux_a = sum([binom(contingency_table[i, i], 2) \
			for i in range(contingency_table.shape[0])])
		aux_aux_a = sum([binom(a_i, 2) for a_i in a_row_sum])
		aux_aux_b = sum([binom(b_j, 2) for b_j in b_col_sum])
		aux_b = aux_aux_a * aux_aux_b / binom(N, 2)
		aux_c = 0.5 * (aux_aux_a + aux_aux_b)
		adjusted_rand_index = (aux_a - aux_b)/(aux_c - aux_b)

		return adjusted_rand_index

	def jackard_index(labels, inst_cluster_id): 
		"""
			The jackard index is pretty much just like
			rand index. However, it does not consider the
			f_00 measure. So,

			jackard_index := f_11 / (f_01 + f_10 + f_11)
		"""
		junk, counters = ClusterMetrics.rand_index(\
			labels, 
			inst_cluster_id,
			return_counters=True)

		# f_00 is not considered, set it to the neutral element
		# of addition, 0.
		counters[int("00", 2)] = 0

		# jackard_index := f_11 / (f_01 + f_10 + f_11)
		jackard_index = counters[int("11", 2)] / sum(counters)

		return jackard_index

	def best_cluster_num(dataset,
		clustering_func,
		k_max, 
		k_min=2,
		metric="silhouette", 
		labels=None,
		full_output=True,
		warnings=True,
		cluster_func_args=None):

		if k_max <= 0 or k_min <= 0:
			if warnings:
				print("Error: \"k_min\"/\"k_max\" must be > 0")
			return None

		if k_min > k_max:
			if warnings:
				print("Error: \"k_min\" must",
					"be <= \"k_max\"")
			return None

		# Parameters checking
		if metric not in {"silhouette", "bss", "sse", "jackard", "rand"}:
			if warnings:
				print("Unknown metric \"" + metric + "\"")
			return None

		if metric in {"jackard", "rand"}:
			# Outter metrics
			if labels is None:
				if warnings:
					print("\"" + metric + "\" need instance \"labels\"",
						"as an Outter Clustering Metric.")
				return None

			args = {
				"inst_cluster_id" : None,
				"labels" : labels,
			}

			if metric == "jackard":
				chosen_metric_func = ClusterMetrics.jackard_index
			else:
				chosen_metric_func = ClusterMetrics.rand_index

		else:
			args = {
				"dataset" : dataset,
				"centers_coord" : None,
				"inst_cluster_id" : None,
			}

			# Inner metrics
			if metric == "silhouette":
				chosen_metric_func = ClusterMetrics.silhouette
			elif metric == "bss":
				chosen_metric_func = ClusterMetrics.bss
			else:
				chosen_metric_func = ClusterMetrics.sse

		if cluster_func_args is None:
			cluster_func_args = {}
		
		metric_array = array([0.0] * (k_max - k_min + 1))

		for k in range(k_min, k_max+1):
			args = {
				**args,
				**clustering_func(\
					dataset=dataset, 
					k=k,
					**cluster_func_args),
			}

			metric_array[k-k_min] = chosen_metric_func(**args)

		ans = {
			"k interval" : [k_min, k_max],
			"k_chosen" : range(k_min, k_max+1)[metric_array.argmax()],
			"all_k_metrics" : {
				k : k_metric \
				for k, k_metric in \
				zip(range(k_min, k_max+1), metric_array)
			},
		}

		if full_output:
			return ans

		return ans["k_chosen"]

"""
	Useful video link: https://youtu.be/pQJrJR_B7Rc (in Portuguese)
	video title: Vídeo 49 - Agrupamento Hierárquico de Dados
	Channel: ML4U
"""

from numpy import array, zeros, inf, median
from pandas import read_csv
from scipy.special import binom

class Hierclus():

	def __euclideandist__(inst_a, inst_b):
		return (sum((inst_a - inst_b)**2.0))**0.5

	def __linksingle__(supergroup_A, supergroup_B):
		# min({dist(a, b) | a e A ^ b e B}
		"""
			The single linkage is fairly simple:
			the distance between the clusters is
			the smallest distance between one
			pair of instance.
		"""
		min_dist = inf
		for inst_a in supergroup_A:
			for inst_b in supergroup_B:
				min_dist = min(min_dist, \
					Hierclus.__euclideandist__(inst_a, inst_b))

		return min_dist	

	def __linkcomplete__(supergroup_A, supergroup_B):
		# max({dist(a, b) | a e A ^ b e B}
		"""
			The complete linkage picks up the maximum
			distance between two pair of points in
			different clusters. 
		"""
		max_dist = 0.0
		for inst_a in supergroup_A:
			for inst_b in supergroup_B:
				max_dist = max(max_dist, \
					Hierclus.__euclideandist__(inst_a, inst_b))

		return max_dist

	def __linkaverage__(supergroup_A, supergroup_B):
		# (|A|*|B|)^{-1} * sum(a e A){sum(b e B){d(a, b)}}
		cardinality = (supergroup_A.shape[0] * supergroup_B.shape[0])

		cum_sum = 0.0
		for inst_a in supergroup_A:
			for inst_b in supergroup_B:
				cum_sum += Hierclus.__euclideandist__(inst_a, inst_b)

		return cum_sum / cardinality

	def __linkmedian__(supergroup_A, supergroup_B):
		# (|A|*|B|)^{-1} * sum(a e A){sum(b e B){d(a, b)}}
		cardinality = (supergroup_A.shape[0] * supergroup_B.shape[0])

		dists = array([0.0] * cardinality)

		counter = 0
		for inst_a in supergroup_A:
			for inst_b in supergroup_B:
				dists[counter] = Hierclus.__euclideandist__(inst_a, inst_b)
				counter += 1

		return median(dists)

	def __choosecriterion__(criterion):
		criterion = criterion.lower()

		if criterion not in {"single", "complete", "average", "median"}:
			print("Error: unrecognized criterion",
				"option \"" + criterion + "\".")
			return None

		elif criterion == "single":
			return Hierclus.__linksingle__

		elif criterion == "complete":
			return Hierclus.__linkcomplete__

		elif criterion == "median":
			return Hierclus.__linkmedian__

		# Default case
		return Hierclus.__linkaverage__

	def run(dataset, criterion="single", min_cluster_num=1):
	
		clustering_metric = Hierclus.__choosecriterion__(criterion)
		if clustering_metric is None:
			return None

		# Cluster hierarchy
		cluster_tree = {i : i \
			for i in range(dataset.shape[0])}
		group_heights = {i : 0.0 \
			for i in range(dataset.shape[0])}

		# This array will keep the greatest supergroup
		# that every instace are in
		outtermost_group_id = array([i \
			for i in range(dataset.shape[0])])

		group_id_counter = dataset.shape[0]

		# Stop criterion: run the algorithm while the number
		# of supergroups (a group that contain smaller groups)
		# is larger than 1. In other words, run the algorithm
		# while there is at least one missing connection between
		# a pair of groups.
		num_supergroups = dataset.shape[0]

		while num_supergroups > min_cluster_num:

			aux_dist = array([None] * int(binom(num_supergroups, 2)))

			supergroup_ids = tuple(cluster_tree.keys())

			k = 0
			for i in range(num_supergroups):
				supergroup_A_id = supergroup_ids[i]
				for j in range(i+1, num_supergroups):
					supergroup_B_id = supergroup_ids[j]
					aux_dist[k] = {
						"clustering_metric" :\
							clustering_metric(\
								dataset[supergroup_A_id == outtermost_group_id,:],\
								dataset[supergroup_B_id == outtermost_group_id,:]),\
						"supergroup_ids" : (supergroup_A_id, supergroup_B_id)}
					k += 1

			# Get the nearest pair of groups based on the
			# user given criterion.
			nearest_pair = min(aux_dist,
				key = lambda k : k["clustering_metric"])
			nearest_supergroup_A_id, nearest_supergroup_B_id = \
				nearest_pair["supergroup_ids"]

			# Build up a new level in the cluster tree, unifying
			# the two nearest groups into a new supergroup.
			supergroup_A = cluster_tree.pop(nearest_supergroup_A_id)
			supergroup_B = cluster_tree.pop(nearest_supergroup_B_id)
			cluster_tree = {
				group_id_counter : {
					nearest_supergroup_A_id : supergroup_A,
					nearest_supergroup_B_id : supergroup_B
				}, 
				**cluster_tree,
			}

			# Update outtermost supergroup indexes
			outtermost_group_id[\
				(outtermost_group_id == nearest_supergroup_A_id) |\
				(outtermost_group_id == nearest_supergroup_B_id) ]\
				= group_id_counter

			# Keep the height of each supergroup in order
			# to construct the dendrogram.
			group_heights[group_id_counter] = \
				nearest_pair["clustering_metric"]

			# Update supergroup counter in order to keep the
			# supergroup ids unique
			group_id_counter += 1
			
			num_supergroups = len(cluster_tree)

		# Build up final answer (cluster tree + nodes heights
		# for dendrogram)
		ans = {
			"cluster_tree" : cluster_tree,
			"group_heights" : group_heights,
			"cluster_num" : len(cluster_tree),
		}

		return ans

if __name__ == "__main__":
	import sys

	if len(sys.argv) < 3:
		print("usage: " + sys.argv[0] + " <data_filepath> <linkage_type>",
			"\t[-label class_label]",
			"\t[-sep data_separator, default to \",\"]",
			"\t[-n minimal_number_of_clusters, default is 1]",
			"\nWhere <linkage_type> must be one of the following:",
			"\tsingle: connect the nearest groups using the nearest pair of instances.",
			"\tcomplete: connect the nearest groups using the farthest pair of instances.",
			"\taverage: connect the nearest groups using an average distance between the instances.", 
			"\tmedian: connect the nearest groups using the median distance between the instances.", 
			sep="\n")
		exit(1)

	try:
		sep = sys.argv[1 + sys.argv.index("-sep")]
	except:
		sep = ","

	dataset = read_csv(sys.argv[1], sep=sep)

	try:
		class_col = sys.argv[1 + sys.argv.index("-label")]
		class_labels = dataset.pop(class_col)
	except:
		class_labels = None

	try:
		n = int(sys.argv[1 + sys.argv.index("-n")])
	except:
		n = 1

	ans = Hierclus.run(
		dataset.iloc[:,:].values, 
		criterion=sys.argv[2],
		min_cluster_num=n)

	if ans is not None:
		print("Dendrogram:")

		def __recursiveprint__(tree, heights, level):
			for key in tree:
				print("->" * level, end=" ")
				if type(tree[key]) == type(dict()):
					print("Height: (" + str(heights[key]) + ")", ":")
					__recursiveprint__(tree[key], heights, level + 1)
				else:
					print("value:", dataset.iloc[key,:].values, "index:", key)
					

		__recursiveprint__(ans["cluster_tree"], ans["group_heights"], 0)

		ans.pop("cluster_tree")
		ans.pop("group_heights")
		print("\nMore information:")
		for item in ans:
			print(item, ":", ans[item])

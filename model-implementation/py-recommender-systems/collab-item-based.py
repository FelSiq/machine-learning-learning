from pandas import DataFrame
import numpy as np
"""
	Item-Based Collaborative Filtering (IBCF)
"""

class Ibcf:
	def __init__(self, x=None, y=None, neighborhood_size=3):
		# Similarity matrix
		self.sim_matrix = None
		self.neighborhood = None
		self.neighborhood_size = -1

		if x is not None and y is not None:
			self.fit(x, y, neighborhood_size)

	def __build_neighborhood(self, neighborhood_size):
		# Neighbordhood of each item is the top
		# "neighbordhood_size" items which has
		# highest similarity with it
		item_labels = self.sim_matrix.columns
		self.neighborhood = {}
		for item_label in item_labels:
			item_label_index = np.where(item_labels == item_label)
			sim_without_cur_item = self.sim_matrix.loc[item_label].values

			# Multiply current item self-similarity to -1.0 in order
			# to prevent itself to be considered its own neighbor
			sim_without_cur_item[item_label_index] *= -1.0

			# Get current item neighbor indexes
			top_similar_items = np.argsort(\
				sim_without_cur_item)[-neighborhood_size:]

			# Set self-similarity back to original value
			sim_without_cur_item[item_label_index] *= -1.0

			# Get current item neighbors
			neighbor_labels = item_labels[top_similar_items]
			self.neighborhood[item_label] = set(neighbor_labels)

		self.neighborhood_size = neighborhood_size

	def fit(self, x, y, neighborhood_size=3):
		"""
			Calculate de similarity matrix
			for the given data

			sim_matrix(i, j) 
				= cos(i, j) 
				= (# intersec(i, j))/(#i**0.5 * #j**0.5)
		"""
		user_indexes = sorted(list(frozenset(x)))
		item_labels = sorted(list(frozenset(y)))
		user_count = len(user_indexes)
		item_count = len(item_labels)

		# Square root of item frequency of each item,
		# used for cosine similarity
		sqr_item_count = np.array([
			sum(y == item)
			for item in item_labels
		])**0.5

		# Hash which keeps all user labels related to each
		# item, as this implementation is a item-based collabo-
		# rative recommender system
		users_of_items_hash = {
			item_index : set(x[np.where(item_labels[item_index] == y)[0]])
			for item_index in range(item_count)
		}

		# Similarity matrix
		self.sim_matrix = np.identity(item_count)

		# Fill up similarity matrix using cosine similarity
		# sim(a, b) = #(a intersection b)/((#a)**0.5 * (#b)**0.5)
		for item_a_index in range(item_count-1):
			users_of_item_a = users_of_items_hash[item_a_index]
			for item_b_index in range(item_a_index + 1, item_count):
				users_of_item_b = users_of_items_hash[item_b_index]

				# cur_sim_val = #(a intersection b)
				cur_sim_val = len(users_of_item_a.intersection(users_of_item_b))

				# cur_sim_val = cur_sim_val / ((#a)**0.5 * (#b)**0.5)
				cur_sim_val /= (sqr_item_count[item_a_index] *\
					sqr_item_count[item_b_index])

				# Keep matrix simmetric for convenience
				self.sim_matrix[item_a_index, item_b_index] = cur_sim_val
				self.sim_matrix[item_b_index, item_a_index] = cur_sim_val

		# Transform into a pandas dataframe to maintain
		# the semantics behind each position
		self.sim_matrix = DataFrame(
			self.sim_matrix, 
			columns=item_labels, 
			index=item_labels)

		# Build up neighborhood for each item
		self.__build_neighborhood(neighborhood_size)

		return self.sim_matrix
	
	def predict(self, query, n=-1, neighborhood_size=3):	
		if self.sim_matrix is None:
			raise Exception("First call \"fit\" method" +
				"before making any predictions")

		if self.neighborhood_size != neighborhood_size:
			self.__build_neighborhood(neighborhood_size)
		
		item_labels = self.sim_matrix.columns

		# Calculate Score for each item in database
		# based on the given query, which represents a
		# new user history
		score = DataFrame([
			sum(self.sim_matrix.loc[item_label, 
				self.neighborhood[item_label].intersection(query)])/\
				sum(self.sim_matrix.loc[item_label, 
					self.neighborhood[item_label]])
			for item_label in item_labels
		], index=item_labels, columns=["Score"])

		score.sort_values(
			by=["Score"], 
			ascending=False, 
			inplace=True)

		if n > 0:
			return score[:n]

		return score

if __name__ == "__main__":
	from pandas import read_csv
	import sys

	if len(sys.argv) < 2:
		print("usage:", sys.argv[0], 
			"<data_filepath>",
			"<query, separated by \"sep\" also>",
			"[-sep, default is \",\"]",
			"[-k neighborhood_size, default is 3]", 
			"[-n show_n_items, default is to show all recommendations]", 
				sep="\n\t")
		exit(1)

	try:
		sep = sys.argv[1 + sys.argv.index("-sep")]
	except:
		sep = ","

	try:
		k = int(sys.argv[1 + sys.argv.index("-k")])
	except:
		k = 3

	try:
		n = int(sys.argv[1 + sys.argv.index("-n")])
	except:
		n = -1

	query = set(sys.argv[2].split(sep))

	data = read_csv(sys.argv[1], 
		sep=sep, 
		names=["Users", "Items"])

	model = Ibcf(
		data.iloc[:, 0].values, 
		data.iloc[:,1].values,
		neighborhood_size=k)

	rec = model.predict(query, n=n)

	print("Similarity matrix:", model.sim_matrix, sep="\n", end="\n\n")
	print("Item neighborhood:")
	for item in model.neighborhood:
		print(item, ":", model.neighborhood[item])
	print("\nQuery:", query, sep="\n", end="\n\n")
	print("Recommendations:", rec, sep="\n")

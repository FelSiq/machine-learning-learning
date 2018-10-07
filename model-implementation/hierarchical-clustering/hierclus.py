from numpy import array
from pandas import read_csv

class Hierclus():

	def __linksingle__(dataset, inst_cluster_id):
		pass

	def __linkcomplete__(dataset, inst_cluster_id):
		pass

	def __linkaverage__(dataset, inst_cluster_id):
		pass

	def run(dataset, criterion="single"):
		if criterion.lower() not in {"single", "complete", "average"}:
			print("Error: unrecognized criterion",
				"option \"" + criterion + "\".")
			return None

		inst_cluster_id = None

		return inst_clluster_id

if __name__ == "__main__":
	import sys

	if len(sys.argv) < 3:
		print("usage: " + sys.argv[0] + " <data_filepath> <linkage_type>",
			"\t[-label class_label] [-sep data_separator, default to \",\"]",
			"\nWhere <linkage_type> must be one of the following:",
			"\tsingle: connect the nearest groups using the nearest pair of instances.",
			"\tcomplete: connect the nearest groups using the farthest pair of instances.",
			"\taverage: connect the nearest groups using an average distance between the instances.", 
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

	ans = Hierclus.run(
		dataset.iloc[:,:].values, 
		criterion=sys.argv[2])

	print("Result:")
	for item in ans:
		print(item, ":", ans[item])


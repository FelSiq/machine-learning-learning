import pandas as pd
import sys

"""
	PCA ("Principal Component Analysis") is a popular technique
	based on linear transformations with eigenvectors. In the
	context of machine learning and data science, it is commonly
	used for feature selection (to reduce data dimension).
"""

class Pca:
	def __init__(self):
		# Last transformation eigen-stuff
		self.eigenvalues=None
		self.eigenvectors=None

	def transform(self, filepath, normalize=True, sep=","):
		dataset = pd.read_csv(filepath, sep=sep)
		
		# Normalize the dataset
		#	a) If columns has the same unity measure:
		#		-> x - mean(x)
		#		-> Calc the covariance matrix
		#	b) If not:
		#		-> (x - mean(x))/sd(s) (remove the measure unity)
		#		-> Calc the correlation matrix

		# Center the dataset values to the origin
		dataset -= dataset.mean()

		# Get transformation matrix
		if normalize:
			dataset /= dataset.std()
			mat = dataset.corr()
		else:
			mat = dataset.cov()

		# Now compute eigen-stuff of computed matrix
		
		

		return dataset

	def feature_selection(self, dataset, eigenvalues=None, information=0.95):
		if not 0 < information <= 1:
			raise ValueError("Information parameter should be in (0, 1].")

		if eigenvalues is None and self.eigenvalues is None:
			raise Exception("No eigenvalues of previous transformation"+\
				" found nor eigenvalues parameter was set.")

		if not isinstance(dataset, pd.DataFrame):
			raise TypeError("Dataset parameter must be a pandas.DataFrame instance.")


"""
	Program driver
"""
if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("usage:", sys.argv[0], "<data filepath> [normalize? (0/1)]")
		exit(1)

	try:
		normalize=bool(int(sys.argv[2]))
	except:
		print("Normalize parameter should be 0 or 1.")
		exit(2)
	
	model=Pca()

	# Return dataset centered at system origin with the
	# covariance/correlation matrix eigenvectors as new system
	# basis
	t1_dataset=model.transform(filepath=sys.argv[1],
		normalize=normalize)

	# New system basis
	eigenvectors=model.eigenvectors
	print("Eigenvectors (new system basis):\n", eigenvectors)
	

	# The columns with smallest eigenvalues are cut off first
	eigenvalues=model.eigenvalues
	print("Eigenvalues (used to feature selection):\n", eigenvalues)

	# Interesting fact: sum(eigenvalues) = sum(variance of all columns)
	print("Sum of eigenvalues is the sum of columns variance:\n",
		sum(eigenvalues), sum(t1_dataset.var()))
	
	# Select features mantaining a minimum of 0.95 of 
	# the dataset information
	t2_dataset=model.feature_selection(t1_dataset, \
		eigenvalues, information=0.95)

	print("New dataset:\n", t2_dataset)

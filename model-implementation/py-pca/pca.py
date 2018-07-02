import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

	def _plot_line(self, intercept, slope, vertical=False):
		axes=plt.gca()
		if not vertical:
			x_vals=np.array(axes.get_xlim())
			y_vals=intercept + slope * x_vals
		else:
			x_vals=np.array([0, 0])
			y_vals=intercept + np.array(axes.get_ylim())
		plt.plot(x_vals, y_vals, '--')

	def transform(self, filepath, normalize=True, sep=",", plot=True):
		dataset = pd.read_csv(filepath, sep=sep)
		
		# Normalize the dataset
		#	a) If columns has the same unity measure:
		#		-> x - mean(x)
		#		-> Calc the covariance matrix
		#	b) If not:
		#		-> (x - mean(x))/sd(s) (remove the measure unity)
		#		-> Calc the correlation matrix

		# Center the dataset values to the origin
		dataset -= dataset.mean(axis=0)

		# Get transformation matrix
		if normalize:
			dataset /= dataset.std(axis=0)
			mat = dataset.corr()
		else:
			mat = dataset.cov()

		# Now compute eigen-stuff of computed matrix
		self.eigenvalues, self.eigenvectors=np.linalg.eig(mat)
		plot &= len(self.eigenvectors) == 2

		if plot:
			plt.subplot(1, 2, 1)
			plt.plot(dataset.iloc[:,0].values, dataset.iloc[:,1].values, "o")
			slope_pca_1=self.eigenvectors[0,1]/self.eigenvectors[0,0]
			slope_pca_2=self.eigenvectors[1,1]/self.eigenvectors[1,0]
			self._plot_line(0, slope_pca_1)
			self._plot_line(0, slope_pca_2)
		else:
			print("Warning: cannot plot with dimensions diff than 2.")

		# Linear transformation from eigenvectors
		# to rotate the data to the eigenvectors basis
		dataset=pd.DataFrame(np.matmul(dataset, self.eigenvectors))
	
		if plot:
			plt.subplot(1, 2, 2)
			plt.plot(dataset.iloc[:,0].values, dataset.iloc[:,1].values, "o")
			self._plot_line(0, 0)
			self._plot_line(0, 0, vertical=True)
			plt.show()

		return dataset

	def feature_selection(self, dataset, eigenvalues=None, information=0.95):
		if not 0 < information <= 1:
			raise ValueError("Information parameter should be in (0, 1].")

		if eigenvalues is None and self.eigenvalues is None:
			raise Exception("No eigenvalues of previous transformation"+\
				" found nor eigenvalues parameter was set.")

		new_dataset=pd.DataFrame.copy(dataset)

		# Normalize eigenvalues to calculate the percentage
		# of information retained while removing less important
		# columns
		e_vals_norm=self.eigenvalues/sum(self.eigenvalues)

		cur_information=1.0
		while cur_information > information:
			break
	
		return new_dataset


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
	print("Sum of eigenvalues is the sum of columns unbiased variance:\n",
		sum(eigenvalues), sum(t1_dataset.var(ddof=1, axis=0)))

	# Select features mantaining a minimum of 0.95 of 
	# the dataset information
	t2_dataset=model.feature_selection(t1_dataset, \
		eigenvalues, information=0.95)

	print("New dataset:\n", t2_dataset)

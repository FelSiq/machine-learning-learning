import pandas as pd
import numpy as np

"""
INFORMATION:
This is a naive bayes classifier non-optmized python implementation.
First call 'fit' method, then you can call 'predict' method whenever you want.
"""

class naiveBayes:
	def __init__(self):
		self.classProbs = dict()
		self.colProbs = np.array([])
		self.classes = np.array([])
		self.queryPropClass = dict()

	def fit(self, x, y):
		# Class probabilities
		self.classFreqs = pd.value_counts(y)
		self.classProbs = dict(self.classFreqs / np.sum(self.classFreqs))
		self.classFreqs = dict(self.classFreqs)
		self.classes = list(classFreqs.keys())

		# Possible values possibilities
		colNum = x.shape[1]
		for c in self.classes:
			curData = x[y == c]
			for j in range(colNum):
				self.colProbs[c][j] = dict(pd.value_counts(curData[:,j]) / self.classFreqs[c])

	def predict(self, query):
		self.queryPropClass = {key : 1.0 for key in self.classes}

		for c in self.classes:
			for i in range(len(query)):
				queryVal = query[i]
				self.queryPropClass[c] *= self.colProbs[c][i][queryVal]

		return max(queryPropClass)

# Program driver
if __name__ == '__main__':

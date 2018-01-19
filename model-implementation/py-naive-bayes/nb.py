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
		self.colProbs = dict()
		self.classes = np.array([])
		self.queryPropClass = dict()

	def fit(self, x, y):
		# Class probabilities
		self.classFreqs = pd.value_counts(y)
		self.classProbs = dict(self.classFreqs / np.sum(self.classFreqs))
		self.classFreqs = dict(self.classFreqs)
		self.classes = list(self.classFreqs.keys())

		# Compute all possibilities
		colNum = x.shape[1]
		for c in self.classes:
			curData = x[y == c]
			self.colProbs[c] = np.array([dict() for k in range(colNum)])
			for j in range(colNum):
				self.colProbs[c][j] = dict(pd.value_counts(curData[:,j]) / self.classFreqs[c])

		return self

	def predict(self, query, crisp=True, normalizeProb=True):
		self.queryPropClass = {key : np.log(self.classProbs[key]) for key in self.classes}

		# Compute probability using Naive Bayes Theorem for every class
		for c in self.classes:
			for i in range(len(query)):
				queryVal = query[i]
				if queryVal in self.colProbs[c][i].keys():
					self.queryPropClass[c] += np.log(self.colProbs[c][i][queryVal])

		self.queryPropClass = {key: np.exp(val) for key, val in self.queryPropClass.items()}

		# Probability normalization
		if (normalizeProb):
			total = sum(self.queryPropClass.values())
			self.queryPropClass = {key:val/total for key, val in self.queryPropClass.items()}
		
		# If crisp is true, only the most probable class will be returned...
		if (crisp):
			return max(self.queryPropClass)
		# If false, return a dictionary with the probability of each class.
		return self.queryPropClass

# Program driver
if __name__ == '__main__':
	dataset = pd.read_csv('tenis.dat', sep=',')

	model = naiveBayes().fit(x = dataset.iloc[:,1:-1].values, y = dataset.iloc[:,-1].values)

	print(model.predict(['ensolarado', 'fria', 'alta', 'forte'], crisp=False))
	print(model.predict(['?', 'quente', 'alta', '?'], crisp=False))
	print(model.predict(['?', 'quente', 'alta', 'medio'], crisp=False))

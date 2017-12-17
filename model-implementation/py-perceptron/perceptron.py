import numpy as np
import pandas as pd
import random

class perceptron:
	def __init__(self):
		self.weights = np.array([])
		self.linearCoefTheta = 0.0

	"""
	Method used on the fit method. Should not be called outside him. 
	"""
	def _updateWeights(self, x, y, trainStep = 0.1):
		constFactor = y - (np.sum(x * self.weights) + self.linearCoefTheta)

		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] + trainStep * (2.0 * x[i] * constFactor)
		self.linearCoefTheta = self.linearCoefTheta + trainStep * (2.0 * constFactor)

		return y - (np.sum(x * self.weights) + self.linearCoefTheta)

	"""
	Uses the 'weights' and 'linearCoefTheta', adjusted by a previous 'fit' operation, to
	predict a output real value of a given query sample.
	"""
	def predict(self, query, outThreshold = 0.5):
		return int((np.sum(query * self.weights) + self.linearCoefTheta) > outThreshold)

	"""
	The Fit method will adjust the 'weights' and the 'linearCoefTheta' of the perceptron.
	It will repeat the training until the sum of output error is smaller than maxError or
	the number of iterations reaches maxIterations. The trainStep parameter tells how much
	each iteration will influence the current weights of the Perceptron.

	Obvious enough, but I'll still gonna say, x is the independent variables of the dataset,
	and y the dependent/output/label/class value.
	"""
	def fit(self, x, y, maxError = 1.0e-15, maxIterations = 500, trainStep = 0.1, showError = False):
		# Initial fit setup
		curError = maxError * 2
		self.weights = np.array([0.0] * x.shape[1])
		for i in range(len(self.weights)):
			self.weights[i] = random.uniform(0.0, 1.0)
		self.linearCoefTheta = random.uniform(0.0, 1.0)
		
		# Perceptron training loop
		curIteration = 0
		while abs(curError) > maxError and curIteration < maxIterations:
			curIteration += 1
			curError = 0.0
			
			for i in range(x.shape[0]):
				curError += self._updateWeights(x[i], y[i], trainStep)
			
			if showError:
				print('I:', curIteration, '\t- curError:', curError)


# Perceptron testing
"""
ASSUME -1 = boolean 0.
"""
if __name__ == '__main__':
	p = perceptron()

	print('AND TESTING:')
	dataset = pd.read_csv('./test/ANDTruthTable.in', sep=' ')
	p.fit(dataset.iloc[:,:-1].values, dataset.iloc[:, -1].values)
	ANDQueries = np.array([
		[-1, -1, -1],
		[-1, +1, -1],
		[+1, +1, +1],
		[-1, +1, +1],
	])

	for a in ANDQueries:
		print('Query:', a, '\nOutput:', p.predict(a))

	print('\nOR TESTING:')
	dataset = pd.read_csv('./test/ORTruthTable.in', sep=' ')
	p.fit(dataset.iloc[:,:-1].values, dataset.iloc[:, -1].values)

	ORQueries = np.array([
		[-1, -1, -1],
		[-1, +1, -1],
		[+1, +1, +1],
		[-1, +1, +1],
	])

	for o in ORQueries:
		print('Query:', o, '\nOutput:', p.predict(o))

	# Perceptron is a linear separator. Because of this, it is
	# incapable of understanding the XOR Truth table.
	print('\nXOR TESTING (!!):')
	dataset = pd.read_csv('./test/XORTruthTable.in', sep=' ')
	p.fit(dataset.iloc[:,:-1].values, dataset.iloc[:, -1].values)

	XORQueries = np.array([
		[-1, -1],
		[-1, +1],
		[+1, +1],
		[-1, +1],
	])

	for x in XORQueries:
		print('Query:', x, '\nOutput:', p.predict(x))
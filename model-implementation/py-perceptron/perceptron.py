import numpy as np
import pandas as pd
import random

class perceptron:
	def __init__(self):
		self.weights = np.array([])

	"""
	Uses the 'weights' and 'linearCoefTheta', adjusted by a previous 'fit' operation, to
	predict a output real value of a given query sample.
	"""
	def predict(self, query, outThreshold = 0.5):
		return int(np.sum(np.concatenate((query, [1.0])) * self.weights) > outThreshold)

	"""
	Method used on the fit method. Should not be called outside him. 
	"""
	def _updateWeights(self, x, y, trainStep = 0.1):
		xConc = np.concatenate((x, [1.0]))

		# Originally the formula was w[i] = w[i] - transStep(-2.0 * xConc[i] * consFactor),
		# but it was simplified.
		constFactor = y - self.predict(x)
		for i in range(len(self.weights)):
			self.weights[i] += trainStep * xConc[i] * constFactor

		return constFactor

	"""
	The Fit method will adjust the 'weights' and the 'linearCoefTheta' of the perceptron.
	It will repeat the training until the sum of output error is smaller than maxError or
	the number of iterations reaches maxIterations. The trainStep parameter tells how much
	each iteration will influence the current weights of the Perceptron.

	Obvious enough, but I'll still gonna say, x is the independent variables of the dataset,
	and y the dependent/output/label/class value.
	"""
	def fit(self, x, y, maxError = 1.0e-04, maxIterations = 500, trainStep = 0.1, showError = False):
		# Initial fit setup
		meanSqrdError = maxError * 2

		self.weights = np.array([0.0] * (x.shape[1] + 1))
		for i in range(len(self.weights)):
			self.weights[i] = random.uniform(-0.5, 0.5)
		
		# Perceptron training loop
		curIteration = 0
		n = x.shape[0]
		while meanSqrdError > maxError and curIteration < maxIterations:
			curIteration += 1
			meanSqrdError = 0.0
			
			for i in range(n):
				meanSqrdError += (self._updateWeights(x[i], y[i], trainStep))**2.0
			meanSqrdError /= n

			if showError:
				print('i:', curIteration, '\t- meanSqrdError:', meanSqrdError)

		print(self.weights)


# Perceptron testing
"""
ASSUME -1 = boolean 0.
"""
if __name__ == '__main__':
	p = perceptron()

	print('AND TESTING:')
	dataset = pd.read_csv('./test/ANDTruthTable.in', sep=' ')
	p.fit(dataset.iloc[:,:-1].values, dataset.iloc[:, -1].values, showError = True)
	ANDQueries = np.array([
		[0, 0, 0],
		[0, 1, 1],
		[1, 1, 1],
		[1, 0, 0],
		[0, 1, 0],
		[0, 1, 1],
	])

	for a in ANDQueries:
		print('Query:', a, '\t\tOutput:', p.predict(a))

	print('\nOR TESTING:')
	dataset = pd.read_csv('./test/ORTruthTable.in', sep=' ')
	p.fit(dataset.iloc[:,:-1].values, dataset.iloc[:, -1].values, showError = True)

	ORQueries = np.array([
		[0, 0, 0],
		[0, 1, 1],
		[1, 1, 1],
		[1, 0, 0],
		[0, 1, 0],
		[0, 1, 1],
	])

	for o in ORQueries:
		print('Query:', o, '\t\tOutput:', p.predict(o))

	# Perceptron is a linear separator. Because of this, it is
	# incapable of understanding the XOR Truth table.
	print('\nXOR TESTING (!!):')
	dataset = pd.read_csv('./test/XORTruthTable.in', sep=' ')
	p.fit(dataset.iloc[:,:-1].values, dataset.iloc[:, -1].values)

	XORQueries = np.array([
		[0, 0],
		[0, 1],
		[1, 1],
		[0, 1],
	])

	for x in XORQueries:
		print('Query:', x, '\t\tOutput:', p.predict(x))
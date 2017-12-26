"""
Information:


"""
import numpy as np
import random
import math

class mlp:
	def __init__(self, eta = 0.2, maxError = 1.0e-0, maxIt = 300, delta = 0.5):
		self.eta = eta
		self.weights = np.array([])
		self.outputNum = 0
		self.hiddenNum = 0

	def _sigmoid(self, x, lamba = 1.0):
		return (1 + math.e**(-lamba * x))**(-1.0)

	def _adjustWeights(self, prediction):
		# Output Layer


		# Hidden Layer

	def predict(self, query):


	def fit(self, dataset, hiddenNum = 2):
		n = dataset.shape[0]
		x = dataset.iloc[:,:-1].values
		y = dataset.iloc[:, -1].values
		self.outputNum = len(set(y))
		self.hiddenNum = hiddenNum

		self.weights = np.array([[0] * self.hiddenNum, [0] * self.outputNum])
		for i in self.hiddenNum:
			self.weights[0, i] = random.uniform(-0.75, 0.75)
		for i in self.outputNum:
			self.weights[1, i] = random.uniform(-0.75, 0.75)

		meanSqrError = self.maxError * 2.0
		curIteration = 0
		while meanSqrError > self.maxError and curIteration < self.maxIt:
			curIteration += 1
			meanSqrError = 0.0

			for i in range(n):
				curPrediction = self.predict(x[i])
				meanSqrError += self._adjustWeights(y, curPrediction)
			meanSqrError /= n


# Program driver
in __name__ == '__main__':
	mlp = mlp()

	mlp.fit()
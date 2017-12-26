"""
Information:


"""
import pandas as pd
import numpy as np
import random
import math

class mlp:
	def __init__(self, stepSize = 0.2, maxError = 1.0e-05, maxIteration = 300, classThreshold = 0.5):
		self.stepSize = 0.2
		self.maxError = 1.0e-05
		self.maxIteration = 300
		self.classThreshold = 0.5
		self.wHidden = np.array([])
		self.wOutput = np.array([])
		self.outputLayerSize = 0
		self.hiddenLayerSize = 0
		self.inputLayerSize = 0

	def _sigmoid(self, x, Lambda = 1.0):
		return (1.0 + math.e**(-Lambda * x))**(-1.0)

	def _adjustWeights(self, x, y):
		# Output layer
		predict = self.predict(x)
		for i in range(self.hiddenLayerSize):
			w[1, i] += self.stepSize * (y - predict) * predict * (1.0 - predict) * ???

		# Hidden layer
		for i in range(self.outputLayerSize):

	def predict(self, query):
		hiddenLayerOutput = np.array([0] * self.hiddenLayerSize)
		for i in range(self.hiddenLayerSize):
			hiddenNet = np.sum(self.wHidden[i] * np.concatenate((query, 1.0)))
			hiddenLayerOutput[i] = this._sigmoid(hiddenNet)

		finalOutput = np.array([0] * self.outputLayerSize)
		for k in range(self.outputLayerSize):
			outputNet = np.sum(self.wOutput[i] * np.concatenate((hiddenLayerOutput[i], 1.0)))
			finalOutput[k] = self._sigmoid(outputNet)

		return finalOutput

	def _initWeights(self):
		for i in range(self.hiddenLayerSize):
			self.wHidden[i] = [random.random() - 0.5 for j in range(self.inputLayerSize)]
		for i in range(self.outputLayerSize):
			self.wOutput[i] = [random.random() - 0.5 for j in range(self.hiddenLayerSize)]

	def fit(self, dataset, hiddenLayerSize = 2):
		n = dataset.shape[0]
		x = dataset.iloc[:,:-1].values
		y = dataset.iloc[:, -1].values

		self.outputLayerSize = math.ceil(math.log(len(set(y)), 2))
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = dataset.shape[1]

		self._initWeights()

		meanSqrError = self.maxError * 2.0
		curIteration = 0
		while meanSqrError > self.maxError and curIteration < self.maxIteration:
			curIteration += 1
			meanSqrError = 0.0

			for i in range(n):
				curPrediction = self.predict(x[i], y[i])
				meanSqrError += self._adjustWeights(y, curPrediction)
			meanSqrError /= n


# Program driver
in __name__ == '__main__':
	dataset = pd.read_csv('./dataset/XOR.dat', sep = ' ')
	mlp = mlp()
	mlp.fit(dataset)

	XORQueries = np.array([
			[1, 0],
			[0, 0],
			[0, 1],
			[1, 1],
		])

	for q in XORQueries:
		print('query:', q, 'result:', mlp.predict(q))
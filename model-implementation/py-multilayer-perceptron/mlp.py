"""
Information:


"""
import pandas as pd
import numpy as np
import random
import math

class mlp:
	def __init__(self, stepSize = 0.2, maxError = 1.0e-5, maxIteration = 300, classThreshold = 0.5):
		self.stepSize = stepSize
		self.maxError = maxError
		self.maxIteration = maxIteration
		self.classThreshold = classThreshold
		self.wHidden = np.array([])
		self.wOutput = np.array([])
		self.outputLayerSize = 0
		self.hiddenLayerSize = 0
		self.inputLayerSize = 0
		self.hiddenLayerOutput = None
		self.predictOutput = None

	"""
	"""
	def _sigmoid(self, x, Lambda = 1.0):
		return (1.0 + math.e**(-Lambda * x))**(-1.0)
	def _d_sigmoid(self, x):
		return self._sigmoid(x) * (1.0 - self._sigmoid(x))

	"""
	Backward/backpropagation step
	"""
	def _adjustWeights(self, x, y):
		predict = self.predict(x)
		error = y - predict

		delta_o = error * self._d_sigmoid(predict)
		delta_h = self._d_sigmoid() * ?

		self.wOutput += self.stepSize * delta_o * np.concatenate((self.hiddenLayerOutput, [1.0]))
		self.wHidden += self.stepSize * delta_h * np.concatenate((x, [1.0]))

		return np.sum(np.power(y - predict, 2.0))

	"""
	Foward step
	"""
	def predict(self, query, Lambda = 1.0):
		self.hiddenLayerOutput = np.array([0.0] * self.hiddenLayerSize)
		for i in range(self.hiddenLayerSize):
			hiddenNet = np.sum(self.wHidden[i] * np.concatenate((query, [1.0])))
			self.hiddenLayerOutput[i] = self._sigmoid(hiddenNet, Lambda)

		self.predictOutput = np.array([0.0] * self.outputLayerSize)
		for k in range(self.outputLayerSize):
			outputNet = np.sum(self.wOutput[k] * np.concatenate((self.hiddenLayerOutput, [1.0])))
			self.predictOutput[k] = self._sigmoid(outputNet, Lambda)

		return self.predictOutput

	"""
	"""
	def _initWeights(self):
		self.wHidden = [[random.random() - 0.5 for __ in range(self.inputLayerSize + 1)] for _ in range(self.hiddenLayerSize)]
		self.wOutput = [[random.random() - 0.5 for __ in range(self.hiddenLayerSize + 1)] for _ in range(self.outputLayerSize)]
	
	"""
	"""
	def fit(self, x, y, hiddenLayerSize = 2):
		n = dataset.shape[0]
		self.inputLayerSize = x.shape[1]
		self.hiddenLayerSize = hiddenLayerSize
		self.outputLayerSize = y.shape[1]

		self._initWeights()

		meanSqrError = self.maxError * 2.0
		curIteration = 0
		while meanSqrError > self.maxError and curIteration < self.maxIteration:
			curIteration += 1
			meanSqrError = 0.0

			for i in range(n):
				meanSqrError += self._adjustWeights(x[i], y[i])
			meanSqrError /= n

			if (curIteration == self.maxIteration):
				print('Warning: reached max iteration number.')


# Program driver
if __name__ == '__main__':
	dataset = pd.read_csv('./dataset/XOR.dat', sep = ' ')
	mlp = mlp(stepSize = 0.05, maxIteration = 1000)
	mlp.fit(x = dataset.iloc[:, :2].values, y = dataset.iloc[:, 2:].values)
	XORQueries = np.array([
			[1, 0],
			[0, 0],
			[0, 1],
			[1, 1],
		])

	for q in XORQueries:
		print('query:', q, 'result:', mlp.predict(q))
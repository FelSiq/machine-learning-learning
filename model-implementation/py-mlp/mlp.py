"""
Information:


"""
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import random
import math

class mlp:
	def __init__(self):
		self.wHidden = np.array([])
		self.wOutput = np.array([])
		self.outputLayerSize = 0
		self.hiddenLayerSize = 0
		self.inputLayerSize = 0
		self.hiddenLayerOutput = None
		self.predictOutput = None
		self.hiddenNet = np.array([])
		self.outputNet = np.array([])

	"""
	"""
	def _sigmoid(self, x, Lambda = 1.0):
		return 1.0/(1.0 + math.e**(-Lambda * x))

	def _d_sigmoid(self, x, Lambda = 1.0):
		sigRes = self._sigmoid(x, Lambda) 
		return sigRes * (1.0 - sigRes)

	"""
	Backward/backpropagation step
	"""
	def _adjustWeights(self, x, y, stepSize, Lambda = 1.0):
		predict = self.predict(x)
		error = y - predict

		delta_o = []
		for k in range(self.outputLayerSize):
			delta_o.append(error[k] * self._d_sigmoid(self.outputNet[k], Lambda))

		delta_h = []
		for j in range(self.hiddenLayerSize):
			aux = np.sum([delta_o[k] * self.wOutput[k,j] for k in range(self.outputLayerSize)])
			delta_h.append(self._d_sigmoid(self.hiddenNet[j], Lambda) * aux)

		for k in range(self.outputLayerSize):
			self.wOutput[k] += stepSize * delta_o[k] * np.concatenate((self.hiddenLayerOutput, [1.0]))
		
		for j in range(self.hiddenLayerSize):
			self.wHidden[j] += stepSize * delta_h[j] * np.concatenate((x, [1.0]))

		return np.sum(np.power(error, 2.0))

	"""
	Foward step
	"""
	def predict(self, query, Lambda = 1.0):
		self.hiddenNet = np.array([np.sum(j * np.concatenate((query, [1.0]))) for j in self.wHidden])
		self.hiddenLayerOutput = np.array([self._sigmoid(i, Lambda) for i in self.hiddenNet])

		self.outputNet = np.array([np.sum(j * np.concatenate((self.hiddenLayerOutput, [1.0]))) for j in self.wOutput])
		self.predictOutput = np.array([self._sigmoid(j, Lambda) for j in self.outputNet])

		return self.predictOutput

	"""
	"""
	def _initWeights(self):
		self.wHidden = np.array([[random.random() - 0.5 
			for __ in range(self.inputLayerSize + 1)] for _ in range(self.hiddenLayerSize)])
		self.wOutput = np.array([[random.random() - 0.5 
			for __ in range(self.hiddenLayerSize + 1)] for _ in range(self.outputLayerSize)])
	
	"""
	"""
	def fit(self, x, y, hiddenLayerSize = 2, showError = False, stepSize = 0.2, 
		maxError = 1.0e-2, maxIteration = 1000, Lambda = 1.0, returnMSEDelta = False):
		n = x.shape[0]
		self.inputLayerSize = x.shape[1]
		self.hiddenLayerSize = hiddenLayerSize
		self.outputLayerSize = y.shape[1]

		self._initWeights()

		MSEdelta = []

		meanSqrError = maxError * 2.0
		curIteration = 0
		while meanSqrError > maxError and curIteration < maxIteration:
			curIteration += 1
			meanSqrError = 0.0

			for i in range(n):
				meanSqrError += self._adjustWeights(x[i], y[i], stepSize, Lambda)
			meanSqrError /= n

			MSEdelta.append(meanSqrError)

			if (showError):
				print('I:','{value:<{fill}}'.format(value = curIteration, fill = 15), 'MSE:', meanSqrError)
			if (curIteration == maxIteration):
				print('Warning: reached max iteration number (' + str(maxIteration) + ').')

		# This vector has the differente of MSE of iteration (t + 1) and t.
		# Its good for plotting.
		if returnMSEDelta:
			for i in range(1, len(MSEdelta)):
				MSEdelta[i - 1] -= MSEdelta[i]
			return MSEdelta[:-1]
		return None

	def plot(self, x, y, dim=250):
		if x.shape[1] > 2:
			print('E: can\'t  plot a dataset on a higher dimension than two.')
			return

		xCoords = []
		yCoords = []
		for s in x:
			xCoords.append(s[0])
			yCoords.append(s[1])

		plt.scatter(xCoords, yCoords)

		xdim = plt.axes().get_xlim()
		ydim = plt.axes().get_ylim()

		xdiff = (xdim[1] * 1.1 - xdim[0] * 1.1)/dim  
		ydiff = (ydim[1] * 1.1 - ydim[0] * 1.1)/dim  
		xpoints = [i * xdiff + xdim[0] * 1.1 for i in range(dim)]
		ypoints = [i * ydiff + ydim[0] * 1.1 for i in range(dim)]

		classes = np.unique(y)
		classesColors = {key : (random.random(), random.random(), random.random()) for key in classes}
		markerColors = {key : (random.random(), random.random(), random.random()) for key in classes}

		matpoints = [[i, j] for j in ypoints for i in xpoints]
		classifications = []
		for sample in matpoints:
			classifications.append(classesColors[int(self.predict(sample) >= 0.5)])

		xpoints = [m[0] for m in matpoints]
		ypoints = [m[1] for m in matpoints]
		plt.scatter(xpoints, ypoints, c=classifications, s=1.0, marker='.')
		plt.scatter(xCoords, yCoords, c=[markerColors[s] for s in y])
		plt.show()

def scale(x):
	colNum = len(x[0])
	minCol = np.array([ math.inf] * colNum)
	maxCol = np.array([-math.inf] * colNum)

	for sample in x:
		minCol = np.array([min(minCol[i], sample[i]) for i in range(colNum)])
		maxCol = np.array([max(maxCol[i], sample[i]) for i in range(colNum)])

	scaledData = np.array([0,0,0,0])
	scaleFactor = 1.0/(maxCol - minCol)

	for sample in x:
		scaledData = np.vstack([scaledData, (sample - minCol) * scaleFactor])

	return scaledData[1:]

import colorama
# Program driver
if __name__ == '__main__':
	mlp = mlp()

	# -------- XOR DATASET ---------------------------------------------
	#"""
	dataset = pd.read_csv('./dataset/XOR.dat', sep = ' ')
	deltaVec = mlp.fit(
		x = dataset.iloc[:, :2].values, 
		y = dataset.iloc[:, 2:].values, 
		showError = True,
		maxIteration=10000,
		stepSize=0.2)
	
	XORQueries = np.array([
			[1, 0],
			[0, 0],
			[0, 1],
			[1, 1],
		])

	for q in XORQueries:
		print('query:', q, 'result:', mlp.predict(q))

	mlp.plot(dataset.iloc[:, :2].values, dataset.iloc[:, -1].values)

	#"""

	# ------- WINE UCI DATASET -----------------------------------------
	"""	
	dataset = pd.read_csv('./dataset/wine.data', sep = ',', 
		header = 0, names = ['Class'] + ['X' + str(i) for i in range(13)])

	fullSet = {i for i in range(dataset.shape[0])}
	trainSet = random.sample(fullSet, round(dataset.shape[0] * 0.6))
	testSet = list(fullSet - set(trainSet))

	x = dataset.iloc[:, 1:]
	normalizedDataset = (x - x.max())/(x.max() - x.min())

	trainData = normalizedDataset.iloc[trainSet].values
	testData = normalizedDataset.iloc[testSet].values

	deltaVec = mlp.fit(
		x = trainData, 
		y = pd.get_dummies(dataset.iloc[trainSet]['Class']).values, 
		hiddenLayerSize = 3,
		showError = True,
		returnMSEDelta = True,
		maxIteration = 300)
	
	correctResults = 0
	trueLabels = pd.get_dummies(dataset.iloc[testSet]['Class']).values
	for i in range(len(testData)):
		wineQuery = testData[i]
		prediction = np.round(mlp.predict(wineQuery))

		checkResult = int(np.sum(prediction - trueLabels[i]) == 0)

		print(colorama.Fore.GREEN if checkResult else colorama.Fore.RED, 
			'query ID:', i, 
			'predict:', prediction,
			'trueLabel:', trueLabels[i],
			colorama.Fore.RESET)

		correctResults += checkResult


	print('Accuracy:', correctResults/len(testData))

	print(deltaVec)

	"""

	# ------ IRIS DATASET -------------------------------------
	"""
	from sklearn import datasets
	iris = datasets.load_iris()

	fullSet = {i for i in range(len(iris.target))}
	trainSet = random.sample(fullSet, round(len(iris.target) * 0.6))
	testSet = list(fullSet - set(trainSet))

	x = scale(iris.data)

	trainData = x[trainSet]
	testData = x[testSet]

	print(pd.get_dummies(iris.target[trainSet]))

	errorVec = mlp.fit(
		x = trainData, 
		y = pd.get_dummies(iris.target[trainSet]).values, 
		hiddenLayerSize = 4,
		showError = True,
		maxError = 0.04,
		returnMSEDelta = False)
	
	correctResults = 0
	trueLabels = pd.get_dummies(iris.target).values
	for i in range(len(testData)):
		irisQuery = testData[i]
		prediction = np.round(mlp.predict(irisQuery))

		checkResult = int(np.sum(prediction - trueLabels[i]) == 0)

		print(colorama.Fore.GREEN if checkResult else colorama.Fore.RED, 
			'query ID:', i, 
			'predict:', prediction,
			'trueLabel:', trueLabels[i],
			colorama.Fore.RESET)

		correctResults += checkResult


	print('Accuracy:', correctResults/len(testData))

	# print(errorVec)
	"""

	# MIT AI 2010 Final Exam tests
	"""
	dataset = pd.read_csv('dataset/5.in')
	mlp.fit(x=dataset.iloc[:, :2].values, y=dataset.iloc[:, 2:].values,showError=True,maxIteration=10000, hiddenLayerSize=3, stepSize=0.25)
	mlp.plot(dataset.iloc[:, :2].values, dataset.iloc[:, -1].values, dim=300)
	"""
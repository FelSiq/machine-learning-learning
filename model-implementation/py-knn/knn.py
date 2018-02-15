import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.cm
import numpy as np
import colorama
import random
import math

class knn:
	def _mahalanobisDist(self, a, b, sInv):
		diffVec = [a[i] - b[i] for i in range(min(len(a), len(b)))]
		return (np.matmul(np.matmul(diffVec, sInv), diffVec))**0.5

	def _cossineSimilarity(self, a, b):
		dotProduct = sum([a[i] * b[i] for i in range(min(len(a), len(b)))])
		normA = sum(i**2.0 for i in a)**0.5
		normB = sum(j**2.0 for j in b)**0.5
		return math.acos(dotProduct / (normA * normB))

	def _hammingDist(self, a, b):
		return sum([a[i] == b[i] for i in range(min(len(a), len(b)))])

	def _manhattanDist(self, a, b):
		return sum([abs(a[i] - b[i]) for i in range(min(len(a), len(b)))])

	def _euclideanDist(self, a, b):
		return sum([(a[i] - b[i])**2.0 for i in range(min(len(a), len(b)))])**0.5

	def _calculateDist(self, a, b, distance=1):
		if distance == 2: # Manhattan
			return self._manhattanDist(a, b)
		elif distance == 3: # Hamming
			# That distance should actually only work well on discrete datasets. 
			return self._hammingDist(a, b)
		elif distance == 4: # Similarity of cossine values between a and b vectors.
			# It actually return the angle between the vectors (not the cossine), to keep
			# the standard of sorting the distance list crescently. It's mainly used on 
			# text classification.
			return self._cossineSimilarity(a, b)
		elif distance == 5:
			# This distance takes in consideration of the variance of each atributte.
			# Atributes with more variance is weighted more.
			return self._mahalanobisDist(a, b, self._sInv)
		else: # Euclidian
			return self._euclideanDist(a, b)

	def predict(self, x, y, query, k=3, distance=1):
		self._sInv = None
		if distance == 5: # Mahalanobis Distance
			self._sInv = np.linalg.inv(np.cov(x, rowvar=False))

		dist = []
		for i in range(len(x)):
			dist.append((self._calculateDist(query, x[i], distance), y[i]))

		nearestLabels = [inst[1] for inst in sorted(dist, key= lambda k : k[0])[:k]]

		freqs = {key : 0 for key in set(y)}
		for n in nearestLabels:
			freqs[n] += 1

		return max(freqs, key = lambda k : freqs[k])

	def plot(self, x, y, k=3, distance=1, dim=250):
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
			classifications.append(classesColors[self.predict(x, y, sample, k, distance)])

		xpoints = [m[0] for m in matpoints]
		ypoints = [m[1] for m in matpoints]
		plt.scatter(xpoints, ypoints, c=classifications, s=1.0, marker='.')
		plt.scatter(xCoords, yCoords, c=[markerColors[s] for s in y])
		plt.show()

import pandas as pd
import numpy as np
import colorama
if __name__ == '__main__':
	"""
	from sklearn import datasets
	iris = datasets.load_iris()
	fullSetIndexes = {i for i in range(len(iris.target))}
	trainSetIndexes = random.sample(fullSetIndexes, round(len(iris.target) * 0.75))
	testSetIndexes = list(fullSetIndexes - set(trainSetIndexes))

	correctResults = 0
	for i in testSetIndexes:
		prediction = knn().predict(iris.data[trainSetIndexes], iris.target[trainSetIndexes], iris.data[i], distance=5)
		checkResult = int((prediction - iris.target[i]) == 0)

		print(colorama.Fore.GREEN if checkResult else colorama.Fore.RED, 
			'query ID:', i, 
			'predict:', prediction,
			'trueLabel:', iris.target[i],
			colorama.Fore.RESET)

		correctResults += checkResult

	print('Accuracy:', correctResults/len(testSetIndexes))

	"""

	# Example from MIT Artifial Intelligence Course 2010 Final Exam (Quiz 3, Problem 1) 
	# In this example, the data must be normalized or else the decision boundary will not work as expected
	dataset = pd.read_csv('datasets/2.in')
	dataset=dataset.sort_values(by=['ID'])
	normalized_attr = (dataset.iloc[:,1:-1] - dataset.mean(numeric_only=True, 
		axis=0))/(dataset.max(numeric_only=True, axis=0) - dataset.min(numeric_only=True, axis=0))
	knn().plot(normalized_attr.values, dataset.iloc[:,-1].values, k=1, distance=1)

	n = normalized_attr.shape[0]
	inst = [True] * n
	for i in range(n):
		inst[i] = False
		label = knn().predict(x=normalized_attr.iloc[inst].values, 
			y=dataset.iloc[inst,-1].values, query=normalized_attr.iloc[i].values, k=1, distance=1)
		inst[i] = True
		trueLabel = dataset.iloc[i,-1]
		print(colorama.Fore.RED if label != trueLabel else colorama.Fore.GREEN, 
			dataset.iloc[i, 0], ': predict:', label, 'true:', trueLabel, colorama.Fore.RESET)
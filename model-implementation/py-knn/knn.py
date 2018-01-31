from sklearn import datasets
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

	def _euclidianDist(self, a, b):
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
			return self._euclidianDist(a, b)

	def predict(self, x, y, query, k=3, distance=1):
		self._sInv = None
		if distance == 5: # Mahalanobis Distance
			self._sInv = np.linalg.inv(np.cov(x, rowvar=False))

		dist = []
		for i in range(len(x)):
			dist.append((self._calculateDist(query, x[i], distance), y[i]))

		nearestLabels = [inst[1] for inst in sorted(dist)[:k]]

		freqs = {key : 0 for key in set(y)}
		for n in nearestLabels:
			freqs[n] += 1

		return max(freqs, key = lambda k : freqs[k])

import pandas as pd
import numpy as np
if __name__ == '__main__':
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
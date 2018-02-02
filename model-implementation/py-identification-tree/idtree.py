from sklearn import datasets
import numpy as np
import pandas
import math

class idtree:
	def __init__(self, type='numerical', thresholds=10):
		self.thresholds = thresholds
		self.tree = {}
		if not type in {'numerical', 'nominal'}:
			print('E: \'type\' must be \'numerical\' or \'nominal\'. Assuming \'numerical\'.')
			self.type = 'numerical'
		else:
			self.type = type

	def _subsetEntropy(subset, base=2):
		probs = np.unique(subset)/len(subset)
		return -sum([probs[i] * math.log(probs[i], base) for i in range(len(probs))])

	def _setEntropy(instSet, base=2):
		totalDisorder = 0.0
		for subset in instSet:
			totalDisorder += self._subsetEntropy(subset, base) * len(subset)/len(instSet)
		return totalDisorder

	def fit(self, x, y, base=2, maxEntropy=0.1, precision=3):
		maxAttribVals = np.max(x, axis=0)
		minAttribVals = np.min(x, axis=0)

		attribThresholds = np.round([[minAttribVals[j] + i * (maxAttribVals[j] - minAttribVals[j])/(self.thresholds + 2) 
			for i in range(self.thresholds + 2)][1:-1] 
			for j in range(len(minAttribVals))], precision)

		instNum = x.shape[0]
		attrNum = x.shape[1]

		usedComb = []
		stack = []
		while something??:
			setEntropies = {}
			for attr in range(attrNum):
				for thrs in range(self.thresholds):
					if not (attr, thrs) in usedComb:
						instSet = something??
						setEntropies[(attr, thrs)] = self._setEntropy(instSet, base)

			# Get min entropy set
			choosenSet = min(setEntropies, key=lambda k : setEntropies[k])
			childLeft = None
			childRight = None
			classLabel = ''

			if setEntropies[choosenSet] <= maxEntropy:
				# This is a new leaf node
				classLabel = something?? 
			else:
				# More filtering should be done, not a leaf node yet.
				childLeft = something??
				childRight = something??
				stack.append(something??)

			self.tree[curNode] = {
				'Attr': choosenSet[0], 
				'Threshold': attribThresholds[choosenSet[0]][choosenSet[1]],
				'childLeft': childLeft,
				'childRight': childRight,
				'classLabel': classLabel}


if __name__ == '__main__':
	iris = datasets.load_iris()


	idtree().fit(iris.data, iris.target)
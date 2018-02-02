from sklearn import datasets
import numpy as np
import pandas
import math

class idtree:
	def __init__(self, type='continuous', thresholds=10):
		self.thresholds = thresholds
		self.tree = {}
		if not type in {'continuous', 'discrete'}:
			print('E: data \'type\' must be \'continuous\' or \'discrete\'. Assuming \'continuous\'.')
			self.type = 'continuous'
		else:
			self.type = type

	def _subsetEntropy(subset, base=2):
		classes, absFreqs = np.unique(subset, return_counts=True)
		probs = absFreqs/len(subset)
		return -sum([probs[i] * math.log(probs[i], base) for i in range(len(probs))])

	def _setEntropy(instSet, base=2):
		totalDisorder = 0.0
		totalSetLen = 0
		for subset in instSet:
			totalSetLen += len(subset)
		for subset in instSet:
			totalDisorder += self._subsetEntropy(subset, base) * len(subset)/totalSetLen
		return totalDisorder

	def fit(self, x, y, base=2, maxEntropy=0.1, precision=3):

		continuousData = (self.type == 'continuous') # Just a little optimization

		if continuousData:
			# Continuous data
			maxAttribVals = np.max(x, axis=0)
			minAttribVals = np.min(x, axis=0)
			attribThresholds = np.round([[minAttribVals[j] + i * (maxAttribVals[j] - minAttribVals[j])/(self.thresholds + 2) 
				for i in range(self.thresholds + 2)][1:-1] 
				for j in range(len(minAttribVals))], precision)
			stack = [{
				'Instances' : [True] * instNum,
				'Attr:': -1, # Just for non-leaf nodes
				'Threshold': -1, # Just for continuous data
				'Childrens': [], # Just for non-leaf nodes
				'ClassLabel': ''}] # Just for leaf nodes
		else:
			# Discrete data
			attribValues = 
			stack = [{
				'Instances' : [True] * instNum,
				'Attr:': -1, # Just for non-leaf nodes
				'ChildrenValues': [], # Just for discrete data
				'Childrens': [], # Just for non-leaf nodes
				'ClassLabel': ''}] # Just for leaf nodes
		instNum = x.shape[0]
		attrNum = x.shape[1]

		usedComb = []

		# Don't necessarily need to be a stack. Any data structure 
		# works, even if unstable (like a heap) or a random sorted array.

		while len(stack):
			curNode = stack.pop()

			if self._setEntropy(y[curNode['Instances']]) > maxEntropy:
				# Not a leaf node
				setEntropies = {}
				for attr in range(attrNum):
					if continuousData:
						# Continuous data approach
						for thrs in range(self.thresholds):
							if not (attr, thrs) in usedComb:
								instSet = something??
								setEntropies[(attr, thrs)] = self._setEntropy(instSet, base)
					else:
						# Discrete data approach
						instSet = something??
						setEntropies[attr] = self._setEntropy(instSet, base)

				if continuousData:
					# Continuous data approach
					# Get min entropy set
					curNode['Attr'], curNode['Threshold'] = min(setEntropies, key=lambda k : setEntropies[k])

					# Generate children nodes
					# Continuous data tree is a binary tree.
					# Implementation detal: left children: less or equal than / right children : greater than
				else:
					# Discrete data approach
					# Get min entropy set
					# Generate children nodes
					# Each Discrete data tree node has one children for each attribute different value 

			else:
				# New leaf node
				classes, counts = np.unique(y[curNode['Instances']], return_counts=True)
				majorityClass = max(zip(classes, counts), key = lambda k : k[1])[0]
				curNode['ClassLabel'] = majorityClass 

if __name__ == '__main__':
	iris = datasets.load_iris()


	idtree().fit(iris.data, iris.target)
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd
import math
import copy
import random

class idtree:
	def __init__(self, type='continuous', thresholds=10):
		self.thresholds = thresholds
		self.tree = {}
		if not type in {'continuous', 'discrete'}:
			print('E: data \'type\' must be \'continuous\' or \'discrete\'. Assuming \'continuous\'.')
			self.type = 'continuous'
		else:
			self.type = type

	def _subsetEntropy(self, subset, base=2):
		classes, absFreqs = np.unique(subset, return_counts=True)
		probs = absFreqs/len(subset)
		return -sum([probs[i] * math.log(probs[i], base) for i in range(len(probs))])

	def _setEntropy(self, instSet, classLabels, base=2):
		totalSetLen = 0
		for subset in instSet:
			totalSetLen += len(subset)
		
		totalDisorder = 0.0
		for subset in instSet:
			totalDisorder += self._subsetEntropy(classLabels[subset], base) * (len(subset)/totalSetLen)

		return totalDisorder

	def _initNode(self, ID=0, Instances=list(), Attr=-1, Threshold=-1.0, 
		ClassLabel='', ChildrenValues=[], Deep=0):
		if self.type == 'continuous':
			return {
				'ID': ID,
				'Instances': Instances,
				'Attr': Attr, # Just for non-leaf nodes
				'Threshold': Threshold, # Just for continuous data
				'ClassLabel': ClassLabel,
				'Deep': Deep}
		else:
			return {
				'ID': ID,
				'Instances': Instances,
				'Attr': Attr, # Just for non-leaf nodes
				'ChildrenValues': ChildrenValues, # Just for discrete data
				'ClassLabel': ClassLabel, 
				'Deep': Deep}

	def _checkPred(self, curNode, target, predVec):
		found = False

		while not found and curNode != -1:
			print(curNode)
			found = curNode == target
			curNode = predVec[curNode]

		return found

	def fit(self, x, y, base=2, maxEntropy=0.1, precision=3, maxDeep=5, showError=False):

		continuousData = (self.type == 'continuous') # Just a little optimization

		instNum = x.shape[0]
		attrNum = x.shape[1]

		if not continuousData:
			# Discrete data
			predVec = {0 : -1}
			attribValues = [set() for i in range(attrNum)]
			for inst in x:
				for i in range(attrNum):
					attribValues[i].add(inst[i])

		# Don't necessarily has to be a stack. Any data structure 
		# works, even if unstable (like a heap) or a random sorted array.
		stack = [self._initNode(Instances=[i for i in range(instNum)])]
		usedComb = set()

		nodeID = 0
		while len(stack):
			curNode = stack.pop()

			if continuousData:
				# Continuous data
				maxAttribVals = np.max(x[curNode['Instances']], axis=0)
				minAttribVals = np.min(x[curNode['Instances']], axis=0)
				attribThresholds = np.round([[minAttribVals[j] + i * (maxAttribVals[j] - minAttribVals[j])/(self.thresholds + 2) 
					for i in range(self.thresholds + 2)][1:-1] 
					for j in range(len(minAttribVals))], precision)

			curEntropy = self._subsetEntropy(y[curNode['Instances']])
			
			if showError:
				print('NodeID:', curNode['ID'], '\tNode entropy:', curEntropy)

			if curEntropy > maxEntropy and curNode['Deep'] <= maxDeep:
				# Not a leaf node
				if continuousData:
					minCombEntropy = {'value': math.inf, 'comb': (-1,-1), 'instSet': []}
				else:
					minAttrEntropy = {'value': math.inf, 'Attr': -1, 'instSet': []}

				for attr in range(attrNum):
					if continuousData:
						# Continuous data approach
						for thrs in range(self.thresholds):
							if not (attr, thrs) in usedComb:
								curThresholdValue = attribThresholds[attr][thrs]
								# The threshold approach only creates two subsets (less-than and greater-or-equal-than)
								instSet = [[], []]

								for instIndex in curNode['Instances']:
									instSet[x[instIndex][attr] >= curThresholdValue].append(instIndex) 

								curCombEntropy = self._setEntropy(instSet, y, base)
								if curCombEntropy < minCombEntropy['value']:
									minCombEntropy['value'] = curCombEntropy
									minCombEntropy['comb'] = (attr, thrs)
									minCombEntropy['instSet'] = copy.deepcopy(instSet)
					else:
						# Discrete data approach
						# I need to keep track of every path on the tree, because it's
						# possible to reuse the same attribute, but not on the same path.
						if not self._checkPred(curNode['Attr'], attr, predVec): 
							instSet = {key : [] for key in attribValues[attr]}

							for instIndex in curNode['Instances']:
								instSet[x[instIndex][attr]].append(instIndex)

							curAttrEntropy = self._setEntropy([instSet[key] for key in instSet], y, base)
							if curAttrEntropy < minAttrEntropy['value']:
								minAttrEntropy['value'] = curAttrEntropy
								minAttrEntropy['Attr'] = attr
								minAttrEntropy['instSet'] = copy.deepcopy(instSet)

				if continuousData:
					# Continuous data approach
					# Get min entropy set
					curNode['Attr'] = minCombEntropy['comb'][0]
					thrsIndex = minCombEntropy['comb'][1]
					instSet = minCombEntropy['instSet']

					# This combination must not be used again
					usedComb.add((curNode['Attr'], thrsIndex))

					# Get true threshold value from possible thresholds generated matrix
					curNode['Threshold'] = attribThresholds[curNode['Attr']][thrsIndex]

					# Generate children nodes
					# Continuous data tree is a binary tree
					self.tree[curNode['ID']] = {'node': curNode, 'lThan': nodeID+1, 'goeThan': nodeID+2}
					stack.append(self._initNode(ID=nodeID+2, Instances=instSet[1], Deep=curNode['Deep']+1))
					stack.append(self._initNode(ID=nodeID+1, Instances=instSet[0], Deep=curNode['Deep']+1))
					nodeID += 2

				else:
					# Discrete data approach
					# Get min entropy set
					curNode['Attr'] = minAttrEntropy['Attr']
					# Generate children nodes
					# Each Discrete data tree node has one children for each attribute different value 
					childrens = {}
					for key in attribValues[curNode['Attr']]:
						nodeID += 1
						predVec[nodeID] = curNode['Attr']
						childrens[key] = nodeID
						stack.append(self._initNode(ID=nodeID, Instances=minAttrEntropy['instSet'][key], Deep=curNode['Deep']+1))

					self.tree[curNode['ID']] = {'node': curNode, 'childrens': childrens}

			else:
				# New leaf node
				classes, counts = np.unique(y[curNode['Instances']], return_counts=True)
				majorityClass = max(zip(classes, counts), key = lambda k : k[1])[0]
				curNode['ClassLabel'] = majorityClass 
				self.tree[curNode['ID']] = {'node': curNode, 'lThan': -1, 'goeThan': -1}
		return self

	def predict(self, query):
		curNode = 0
		while self.tree[curNode]['node']['ClassLabel'] == '':
			if self.type == 'continuous':
				if query[self.tree[curNode]['node']['Attr']] < self.tree[curNode]['node']['Threshold']:
					curNode = self.tree[curNode]['lThan']
				else:
					curNode = self.tree[curNode]['goeThan']
			else:
				for value in self.tree[curNode]['childrens']: 
					if query[self.tree[curNode]['node']['Attr']] == value:
						curNode = self.tree[curNode]['childrens'][value]
		return self.tree[curNode]['node']['ClassLabel']

	def plot(self):
		None

if __name__ == '__main__':
	# IRIS TESTING
	"""	
	iris = datasets.load_iris()

	cvfolds = 10
	folds = np.array([random.randint(0, cvfolds-1) for i in range(len(iris.target))])

	accuracies = [0.0] * cvfolds
	for f in range(cvfolds):
		model = idtree().fit(iris.data[folds!=f], iris.target[folds!=f], maxDeep=3)
		correctResults = 0

		testData = iris.data[folds==f]
		testLabels = iris.target[folds==f]

		for i in range(len(testData)):
			label = model.predict(testData[i])
			correctResults += label == testLabels[i]

		accuracies[f] = correctResults/len(testLabels)

	print('CV accuracy: ', np.mean(accuracies))

	model = idtree().fit(iris.data, iris.target, maxDeep=3)
	"""

	# TENIS DATASET
	dataset = pd.read_csv('tenis.dat')

	# LOOCV
	cvfolds = dataset.shape[0]
	folds = np.array([i for i in range(cvfolds)])

	accuracies = [0.0] * cvfolds
	for f in range(cvfolds):
		model = idtree(type='discrete').fit(dataset.iloc[folds!=f,1:-1].values, dataset.iloc[folds!=f, -1].values, maxDeep=10)
		correctResults = 0

		testData = dataset.iloc[folds==f,1:-1].values
		testLabels = dataset.iloc[folds==f, -1].values

		if len(testLabels):
			for i in range(len(testData)):
				label = model.predict(testData[i])
				correctResults += label == testLabels[i]

			accuracies[f] = correctResults/len(testLabels)

	print('CV accuracy: ', np.mean(accuracies))

	model.plot()
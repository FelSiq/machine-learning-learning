from sklearn.preprocessing import LabelEncoder as lbe
import pandas as pd
import random as rd
import numpy as np
import copy

def dist(v1, v2):
	return np.sum(np.power(v1 - v2, 2))**0.5

def matrixEquality(m1, m2, delta = 1.0e-10):
	d = 0
	for i in range(len(m1)):
		d += dist(m1[i], m2[i])
	print('Current total error:', d)
	return d < delta


def kmeans(dataset, k = 3, delta = 1.0e-10, maxit = 300):
	classLabels = [0] * len(dataset)
	itNum = 0

	initCenters = [0] * k
	while len(set(initCenters)) != k:
		for i in range(k):
			initCenters[i] = rd.randint(0, len(dataset) - 1)

	centerCoord = list()
	for i in range(k):
		centerCoord.append(dataset.iloc[initCenters[i]].values)
	centerCoord = np.matrix(centerCoord)

	prevCoords = copy.deepcopy(centerCoord) + delta * 2
	while not matrixEquality(prevCoords, centerCoord, delta) and itNum < maxit:
		itNum += 1

		# Calculate the distances of each example from all centers,
		# and update the class label of each example
		for i in range(len(dataset)):
			newClassLabel = classLabels[i]
			d = [0] * k
			for j in range(k):
				d[j] = dist(dataset.iloc[i].values, centerCoord[j]) 
			classLabels[i] = d.index(min(d))

		# update the centers based on the new class labels of each example
		prevCoords = copy.deepcopy(centerCoord)
		centerCoord *= 0
		numExamples = [0] * k
		for i in range(len(dataset)):
			centerCoord[classLabels[i]] += dataset.iloc[i].values
			numExamples[classLabels[i]] += 1

		for i in range(k):
			if numExamples[i]:
				centerCoord[i] /= numExamples[i]

	print('Total # of iterations:', itNum)

	return np.array(classLabels)


if __name__ == '__main__':
	dataset = pd.read_csv('/home/felipe/Documentos/irisShuffled.in')

	seq = dataset['#'].values
	classTrueVals = dataset['Species'].values
	dataset.drop(columns = ['#', 'Species'], inplace = True)

	classLabels = kmeans(dataset)
	print(classLabels[seq-1])

	# the lbe.fit_transform is not supposed to guess what number
	# corresponds to what class. It's necessary a data analyst to
	# check the results.
	print(lbe().fit_transform(classTrueVals[seq-1]))


	"""
	Got results:

	vec1 = np.array([2, 2, 0, 0, 0, 0, 2, 2, 0, 2, 0, 1, 0, 0, 0, 2, 2, 0, 0, 0, 2, 1, 0, 0, 1, 2, 1, 1, 0, 1, 1, 2, 2, 1, 2, 2, 0,
	 1, 2, 0, 2, 1, 0, 1, 1, 2, 0, 2, 1, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2,
	 1, 2, 2, 2, 0, 0, 2, 2, 1, 2, 2, 0, 1, 0, 1, 0, 1, 2, 2, 2, 1, 1, 2, 1, 0, 0, 1, 2, 0, 2, 2, 0, 1, 0, 1, 2, 0,
	 1, 1, 1, 1, 2, 0, 1, 0, 2, 2, 2, 0, 1, 0, 0, 2, 2, 2, 1, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 0,
	 0, 2])
	vec2 = np.array([2, 2, 0, 0, 0, 0, 2, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 0, 0, 0, 2, 1, 0, 0, 1, 2, 1, 1, 0, 1, 1, 2, 2, 1, 2, 2, 0,
	 1, 2, 0, 2, 1, 0, 1, 1, 2, 0, 2, 1, 2, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2, 2, 2,
	 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 1, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 2, 0,
	 1, 2, 1, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 0, 0, 1, 1, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 0, 2, 1, 2, 0, 0, 0,
	 0, 2])

	sum(vec1 == vec2)/len(vec1)
	# output: 0.88666666666666671 (this is roughly a 88.67% win rate)
	"""
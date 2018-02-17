from sklearn.preprocessing import LabelEncoder as le
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

# Dataset font: http://archive.ics.uci.edu/ml/machine-learning-databases/00368/

def dist(x1, x2):
	return sum(np.power(x1 - x2, 2.0))**0.5 

def weigth(x1, x2, sigma=1.0):
	return math.e**(-0.5 * (dist(x1, x2)/sigma)**2)

def sign(x):
	return 1.0 if x >= 0.0 else -1.0

def dwnn(x, y, query, sigma=1.0, classification=False):
	n = len(y)

	weights = np.array([0.0] * n)
	for i in range(n):
		weights[i] = weigth(query, x[i], sigma)

	res = sum(y * weights)/sum(weights)

	return sign(res) if classification else res

def plot(x, y, dim=250, sigma=1.0, classification=False):
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
		classifications.append(classesColors[dwnn(x, y, sample, sigma, classification)])

	xpoints = [m[0] for m in matpoints]
	ypoints = [m[1] for m in matpoints]
	plt.scatter(xpoints, ypoints, c=classifications, s=1.0, marker='.')
	plt.scatter(xCoords, yCoords, c=[markerColors[s] for s in y])
	plt.show()

if __name__ == '__main__':

	# Test 1:
	"""
	dataset = np.array([[i, i] for i in range(10)])
	query = np.array([7.5])
	out = dwnn(dataset[:, 0], dataset[:, 1], query, 0.2)
	print('Regression result:', out)
	"""
	
	# Test 2:
	"""
	dataset = pd.read_csv('./dataset/dataset_facebook.csv', sep = ';')
	dataset.dropna(axis = 0, inplace = True)
	
	dataset['Type'] = le().fit_transform(dataset['Type'])
	"""

	# Dataset example #1
	"""
	query = np.array([0, 2, 12, 4, 3, 0, 2752, 5091, 178, 109, 159, 3078, 1640, 119, 4, 79, 17, 100]) * 1.0

	out = dwnn(dataset.iloc[:, 1:].values, dataset['Page total likes'].values, query, sigma = 1.0e+2)
	print('Regression result:', out)
	"""

	# Classification test
	dataset = pd.read_csv('dataset/1.in')
	plot(x=dataset.iloc[:,:-1].values, y=dataset.iloc[:,-1].values, classification=True)

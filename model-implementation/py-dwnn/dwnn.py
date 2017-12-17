from sklearn.preprocessing import LabelEncoder as le
import pandas as pd
import numpy as np
import math

# Dataset font: http://archive.ics.uci.edu/ml/machine-learning-databases/00368/

def dist(x1, x2):
	return sum(np.power(x1 - x2, 2.0))**0.5 

def weigth(x1, x2, sigma = 1.0):
	return math.e**(-0.5 * (dist(x1, x2)/sigma)**2)

def dwnn(x, y, query, sigma = 1.0):
	n = len(y)

	weights = np.array([0.0] * n)
	for i in range(n):
		weights[i] = weigth(query, x[i], sigma)

	return sum(y * weights)/sum(weights)

if __name__ == '__main__':

	# Test 1:		
	dataset = np.array([[i, i] for i in range(10)])
	query = np.array([7.5])
	out = dwnn(dataset[:, 0], dataset[:, 1], query, 0.2)
	print('Regression result:', out)
	
	# Test 2:
	dataset = pd.read_csv('./dataset/dataset_facebook.csv', sep = ';')
	dataset.dropna(axis = 0, inplace = True)
	
	dataset['Type'] = le().fit_transform(dataset['Type'])

	# Dataset example #1
	query = np.array([0, 2, 12, 4, 3, 0, 2752, 5091, 178, 109, 159, 3078, 1640, 119, 4, 79, 17, 100]) * 1.0

	out = dwnn(dataset.iloc[:, 1:].values, dataset['Page total likes'].values, query, sigma = 1.0e+2)
	print('Regression result:', out)
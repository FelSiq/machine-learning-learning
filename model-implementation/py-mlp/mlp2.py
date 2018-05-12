import numpy as np

"""
This is a implementation of the Multilayer Perceptron (MLP), a
very popular machine learning algorithm used for classification tasks.
"""

class Mlp:
	def __init__(self, sizes=None):
		self.weights = None

		"""
		User can specify the sizes of the MLP architecture
		directly from the MLP instantiation.
		"""
		if sizes:
			self.build(sizes)

	def build(self, sizes=(2,2,2)):
		"""
		Builds up the Multilayer Perceptron architecture.
		The sizes tuple is built following the order:
		#1 : # of input layer neurons
		#2 : # of hidden layer neurons
		#3 : # of output layer neurons
		"""
		n, m, k = sizes
		self.weights={
			'hidden': np.array([np.random.rand(n+1) - 0.5 
				for i in range(m)]),
			'output': np.array([np.random.rand(m+1) - 0.5 
				for i in range(k)])
		}

		return self.weights

	def sig(self, netVector):
		"""
		Activation function (sigmoid).
		"""
		try:
			return np.array([1.0/(1.0 + np.exp(-x)) 
				for x in netVector])
		except:
			return np.array([1.0/(1.0 + np.exp(-netVector))])	

	def dsig(self, netVector):
		"""
		Activation function (sigmoid) derivative
		"""
		aux=self.sig(netVector)
		return np.array([s * (1.0 - s) for s in aux])

	def foward(self, x, retComplete=False):
		"""
		Foward phase of the MLP
		"""
		hnet = np.dot(self.weights['hidden'], 
			np.concatenate((x, [1.0])))
		fhnet = self.sig(hnet)

		onet = np.dot(self.weights['output'], 
			np.concatenate((fhnet, [1.0])))
		fonet = self.sig(onet)
		
		if retComplete:
			return fonet, onet, fhnet, hnet
		return fonet

	def backward(self, fonet, onet, 
		fhnet, hnet, itErr, x, eta=0.1):
		"""
		Build up Output Layer adjust matrix
		"""
		deltaOut = itErr * self.dsig(onet)
		outputAdjust = np.concatenate((self.sig(hnet), 
			[1.0])) * deltaOut
		"""
		Build up the Hidden Layer adjust matrix
		"""
		hiddenLayerSize = len(self.weights['hidden'])
		inputLayerSize = len(x)	
		xcat = np.concatenate((x, [1.0]))
		hiddenAdjust = [[0.0] * (1 + inputLayerSize) 
			for i in range(hiddenLayerSize)]
		for i in range(hiddenLayerSize):
			auxSum = (np.dot(deltaOut, 
				self.weights['output'][:,i]) * 
				self.dsig(hnet[i])).item()
			for j in range(1+inputLayerSize):
				hiddenAdjust[i][j] = auxSum * xcat[j]
		hiddenAdjust = np.array(hiddenAdjust)
		"""
		Finally, adjust the weights (and thetas)
		"""
		self.weights['output'] += eta * outputAdjust
		self.weights['hidden'] += eta * hiddenAdjust

	def train(self, x, y, m=2, maxError=1.0e-3, maxIt=1e+4):
		"""
		A typical train of the MLP consists of the following pseudo-code:
		WHILE MeanSquaredError > maxError and it < maxIt DO:
			FOR each instance (x, y) at Train Set DO:
				metadata <- foward(x) (feed the MLP with 
					vector x attributes)
				error <- calcError(y, metadata$MLPOutput)
				backwards(metadata, error) (adjust MLP weights)
				MeanSquaredError <- MeanSquaredError + error^2
			END FOR
			MeanSquaredError <- MeanSquaredError / # of 
				instances of Train Set
		END WHILE
		"""

		"""
		If build(sizes) method was not called explicity by the user,
		try to build up a equivalent architecture using some train set
		dimensions.
		"""
		if self.weights == None:
			self.build((x.shape[1], m, y.shape[1]))

		meanSqrError = maxError+1.0	# MSE (stop condition) 
		itToPrint = max(1, maxIt//100)	# Iterations between each MSE print
		instNum = x.shape[0]		# Number of instances of the train set
		i = 0				# Iteration counter (stop condition)

		while meanSqrError > maxError and i < maxIt:
			i += 1
			meanSqrError = 0.0
			for inst, lab in zip(x, y):
				fonet, onet, fhnet, hnet = self.foward(inst, 
					retComplete=True)
				"""
				Calculate iteration error (value used on the
				Backward phase.
				"""
				itErr = np.array(list(lab - fonet))

				"""
				Call the backward phase, using foward's 
				information.
				"""
				self.backward(fonet, onet, 
					fhnet, hnet, itErr, inst)

				"""
				Sum up to the SE and pick the next train set
				instance.
				"""
				meanSqrError += sum(itErr**2.0)
			
			"""
			Divide the sum of SEs to the number of train set 
			instances, so it will be a MSE.
			"""
			meanSqrError /= instNum

			if i % itToPrint == 0:
				print(i, ':', meanSqrError)
	def print(self):
		if self.weights:
			print('Output Layer weights:',
				self.weights['output'],
				'Hidden Layer weights',
				self.weights['hidden'], 
				sep='\n')
		else:
			print('This MLP has no architecture.')

"""
Program Driver for inner testing
"""
if __name__ == '__main__':
	import pandas as pd
	m = Mlp()

	m.print()
	
	dataset = pd.read_csv('dataset/XOR.dat', sep=' ')
	X = dataset.iloc[:,:-1].values
	Y = dataset.iloc[:,-1].values
	Y = np.reshape(Y, (len(Y), 1))
	print(Y)

	m.train(X, Y, maxIt=np.inf)

	for s, l in zip(X, Y):
		print('Expected output:', l,'\tMLP output:', m.foward(s))

from numpy import concatenate as concat
from numpy import transpose as transp
from numpy import matrix
from numpy import random
from numpy import array
from numpy import exp
from numpy import matmul
from numpy import dot

class Mlp:
	def __init__(self):
		self.weights = None

	def build(self, sizes=(2,2,2)):
		"""
		Builds up the Multilayer Perceptron architecture.
		The sizes tuple is built following the order:
		#1 : # of input layer neurons
		#2 : # of hidden layer neurons
		#3 : # of output layer neurons
		"""
		n, m, k = sizes
		self.weights=[]
		self.weights.append(matrix([random.rand(n+1) - 0.5 for i in range(m)]))
		self.weights.append(matrix([random.rand(m+1) - 0.5 for i in range(k)]))
		self.weights=array(self.weights)
		return self.weights

	def sig(self, x):
		return 1.0/(1.0 + exp(-x))

	def dsig(self, x):
		s=sig(x)
		return s * (1.0 - s)

	def foward(self, x):
		hnet = matmul(self.weights[0], concat((x, [1.0]))).tolist()[0]
		fhnet = [self.sig(x) for x in hnet]

		onet = matmul(self.weights[1], concat((fhnet, [1.0]))).tolist()[0]
		fonet = [self.sig(x) for x in onet]

		return fonet, onet, fhnet, hnet

	def backward(self, fonet, onet, fhnet, hnet, itErr, x, eta=0.1):
		deltaOut = itErr * [dsig(x) for x in onet]
		outputAdj = [f*d for d, f in zip(concat((fhnet, [1.0])), deltaOut)]

		deltaHid = [self.dsig(hnet[c]) * 
			dot(deltaOut, self.weights[0][c]) 
			for c in range(self.weights[0].shape[0])]
		hiddenAdj = [i*d for i, d in zip(concat((x, [1.0])), deltaHid)]

		self.weights[0] += eta * outputAdj
		self.weights[1] += eta * hiddenAdj

	def train(self, x, y, m=2, maxError=1.0e-4, maxIt=1e+4):
		if self.weights == None:
			self.build((x.shape[1], m, y.shape[1]))

		i = 0
		meanSqrError = 0.0
		itToPrint = max(1, maxIt//100)
		instNum = x.shape[0]
		while error > maxError and i < maxIt:
			i += 1
			error = 0.0
			for inst, lab in zip(x, y):
				fonet, onet, fhnet, hnet = self.foward(inst)
				itErr = (lab - out)
				self.backward(fonet, onet, fhnet, hnet, itErr, inst)
				meanSqrError += itErr**2.0
			
			error /= InstNum
			if not i % itToPrint:
				print(i, ':', meanSqrError)

if __name__ == '__main__':
	m = Mlp()	
	print(m.build((2, 3, 1)))

	print(m.foward([1.0, 1.0]))
	

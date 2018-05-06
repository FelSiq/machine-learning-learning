from numpy import matrix
import math

"""
This module is dedicated of implementing a bidimensional matrix pooler.
"""
class Pooling:
	def __init__(self, stride=(1,1), mask=(2,2), maxPooling=True):
		"""
		stride:	(horizontal step, vertical step)
		mask:	(width, height)

		All values these will be 1 by default if given 0 or less,
		as you can see below.
		"""
		self.vertStride = max(1, stride[1])
		self.horizStride = max(1, stride[0])
		self.widthMask = max(1, mask[0])
		self.heightMask =  max(1, mask[1])

		# If false, the min pooling will be used instead
		self.maxPooling = maxPooling

	def pooling(self, mat):
		m = math.ceil(mat.shape[1]/self.horizStride)
		n = math.ceil(mat.shape[0]/self.vertStride)
		retMatrix = matrix([[0.0] * m for i in range(n)])
		for i in range(n):
			for j in range(m):
				aux = mat[
					(i*self.vertStride):(i*self.vertStride + self.widthMask), 
					(j*self.horizStride):(j*self.horizStride + self.heightMask)]
				retMatrix[i, j] = aux.max() if self.maxPooling else aux.min()
		return retMatrix

"""
Program Driver for internal testing.
"""
if __name__ == '__main__':
	m=matrix([
		[.77, -.11, .11, .33, .55, -.11, .33],
		[-.11, 1.0, -.11, .33, -.11, .11, -.11],
		[.11, -.11, 1.0, -.33, .11, -.11, -.11],
		[.33, .33, -.33, .55, -.33, .33, .33],
		[.55, -.11, .11, -.33, 1.0, -.11, .11],
		[-.11, .11, -.11, .33, -.11, 1.0, -.11],
		[.33, -.11, .55, .33, .11, -.11, .77]
	])

	r=Pooling(stride=(1,2), mask=(4, 4)).pooling(m)
	print(r)

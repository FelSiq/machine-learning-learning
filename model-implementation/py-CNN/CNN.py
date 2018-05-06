import relu
import convolution
import pooling

"""
This code implements a cute version of a Convolutional Neural Network (CNN).
It's not supposed to be efficient nor have any special new implementation 
technique, it's just for study purposes.
"""

class CNN:
	def __init__(self):
		"""
		"""
		self.Architecture=[]
	
	"""
	Note: all methods that starts with "cl" stands for "Create 
	Layer" method.
	"""

	def clConvolution(self, size, mask):
		"""
		Creates a Convolutional Layer. Size is the number of neurons.
		Mask is the kernel matrix that will be used during the matrix
		convolution process.
		"""

	def clPooling(self, size, stride=(1,1), mask=(3,3), maxPool=True):
		"""
		Creates a Pooling Layer. Default is max(imum) Pooling, which
		yields the maximum value beneath the pooling mask. The stride
		should be a tuple defined as (horizontal stride, vertical 
		stride). Size is the number of neurons inside it. The mask
		tuple is (width, height) of the pooling mask.
		"""

	def clRelu(self, size):
		"""
		"ReLU" stands for "REctified Linear Units". This type of layer
		just appliesa ReLU function f, defined as
		
		f(x) =	x, 	if x > 0
			0,	otherwise.

		This is used necessarily to avoid negative values in filtered
		images.
		"""

"""
Program Driver, just for internal test purposes.
"""
if __name__ == '__main__':

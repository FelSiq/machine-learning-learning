import copy
"""
This module is dedicated of implementing the ReLU function, which is defined as:
f(x) =	x, 	if x > 0
	0, 	otherwise.
"""


class Relu:
    def relu(self, matrix, inplace=False):
        refMat = matrix
        if not inplace:
            refMat = copy.deepcopy(matrix)
        refMat[refMat < 0.0] = 0.0
        if not inplace:
            return refMat


"""
Program Driver for intern testing.
"""
if __name__ == '__main__':
    import random
    from numpy import matrix
    m = matrix([[random.uniform(-1, 1) for j in range(5)] for i in range(5)])
    print('Random inited matrix:\n', m)
    m2 = Relu().relu(m, inplace=False)
    print('Should be equal of prev print (not in-place transformation):\n', m)
    print('Relu-ed matrix:\n', m2)

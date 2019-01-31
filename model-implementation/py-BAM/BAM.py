import sys
import numpy as np
import copy
"""
This is a very simple implementation of a BAM Network
(Bidirectional Associative Memory). It can be used for
error correction on data inside a Hamming Space (binary values).
"""


class BAM:
    def __init__(self, x=None, y=None):
        self.covMatrix = None
        if x is not None and y is not None:
            self.train(x, y)

    def fixVector(self, v):
        # Tranform a vector space to a Hamming (binary) space
        # Given a vector v...
        # v[i]	=	-1, if v[i] < 0.0
        #		+1, if v[i] > 0.0
        #		random({-1, 1}) otherwise.

        for i in range(len(v)):
            if v[i] != 0:
                v[i] = -1.0 if v[i] < 0.0 else 1.0
            else:
                v[i] = np.random.choice([-1.0, 1.0])
        return v

    def train(self, X, Y):
        # Creates BAM covariance matrix
        self.covMatrix = np.dot(np.transpose(X), Y)

    """
	Feed up BAM Network and produce a output based on its
	(memory) covariance matrix, created via "train" method.
	"""

    def feed(self, x):
        Hold = 0.0
        Hcur = 0.0
        var = 1.0
        i = 0
        y = None

        if self.covMatrix is None:
            raise Exception('Method \"train\" must be called before ' +\
             '\"feed\" method.')

        vi = self.fixVector(copy.copy(x))
        while var:
            # From	->	By means of	->	Transform to
            # x_i 	-> 	BAM COV MAT	->	y_i
            y = np.dot(vi, self.covMatrix)
            y = self.fixVector(y)

            # Calc Entropy_i
            # Note that this is a estimation of the Shannon Entropy
            Hcur = -np.dot(vi, np.dot(self.covMatrix, y))

            # Go back to 	<-	By means of	<-	From
            # x_(i+1)	<-	BAM COV MAT	<-	y_i
            vi = np.dot(self.covMatrix, np.transpose(y))
            vi = self.fixVector(vi)

            # Calc deltaH = |H_(i+1) - H_i|
            var = abs(Hcur - Hold)

            # Print current Entropy
            print(i, ':', Hcur, '(diff:', var, ')')
            i += 1

            # Update old entropy and repeat til
            # var is zero.
            Hold = Hcur

        # Return last value of BAM output, y_n
        return y


"""
Program driver
"""
if __name__ == '__main__':
    import pandas
    dataset = pandas.read_csv('tests/dataset0.dat')
    model = BAM(dataset.iloc[:, :10].values, dataset.iloc[:, 10:].values)

    print(model.covMatrix)

    p = dataset.iloc[1, :10].values
    p[0] = -p[0]
    print('result:', model.feed(p))
    print('expected:', dataset.iloc[1:, 10:].values)

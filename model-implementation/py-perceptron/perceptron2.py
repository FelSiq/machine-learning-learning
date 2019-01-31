from numpy import random as rd
from numpy import concatenate as concat
from numpy import dot


class Perceptron2:
    def __init__(self):
        self.weights = None

    def _startWeights(self, n):
        self.weights = rd.rand(n) - 0.5
        print(self.weights)

    def feed(self, query, delta=0.5):
        output = dot(concat((query, [1.0])), self.weights)
        return 1.0 if output >= delta else 0.0

    def train(self, x, y, eta=0.1, maxerr=1.0e-5, maxit=1e+4, printErr=True):
        numattr = x.shape[1]
        self._startWeights(numattr + 1)

        i = 0
        err = maxerr + 1
        errItPrint = max(1, maxit / 1000)
        while i < maxit and err > maxerr:
            i += 1
            err = 0.0
            for inst, label in zip(x, y):
                itErr = label - self.feed(inst)
                self.weights += eta * itErr * concat((inst, [1.0]))
                err += itErr**2.0
            if printErr and not (i % errItPrint):
                print(i, ':', err)
        print('Process finished @ iteration', i, 'with error', err)
        return self.weights


if __name__ == '__main__':
    model = Perceptron2()
    import pandas as pd
    dataset = pd.read_csv('./test/ANDTruthTable.in', sep=' ')

    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    model.train(X, Y, eta=0.001)

    for inst, label in zip(X, Y):
        output = model.feed(inst)
        print(output - label)

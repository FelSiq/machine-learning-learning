import pandas as pd
import numpy as np
import math


class AENN:
    def __init__(self):
        self.removedIndexes = set()
        self.cleanx = None
        self.cleany = None

    def _dist(self, a, b):
        return np.sum(np.power(a - b, 2.0))**0.5

    def _scale(self, x):
        colNum = len(x[0])
        minCol = np.array([math.inf] * colNum)
        maxCol = np.array([-math.inf] * colNum)

        for sample in x:
            minCol = np.array(
                [min(minCol[i], sample[i]) for i in range(colNum)])
            maxCol = np.array(
                [max(maxCol[i], sample[i]) for i in range(colNum)])

        scaledData = np.array([0, 0, 0, 0])
        scaleFactor = 1.0 / (maxCol - minCol)

        for sample in x:
            scaledData = np.vstack(
                [scaledData, (sample - minCol) * scaleFactor])

        return scaledData[1:]

    def enn(self, x, y, k=5, scale=False):
        noise = set()

        if (scale):
            x = self._scale(x)

        n = len(y)
        for i in range(n):
            dist = []
            for j in range(n):
                dist.append(self._dist(x[i], x[j]))

            indexes = [i for i in range(n)]
            dist, indexes = zip(*sorted(zip(dist, indexes)))

            equalNeighbors = sum([y[m] == y[i] for m in indexes[1:(k + 1)]])

            if equalNeighbors < k / 2.0:
                noise = noise.union({i})

        return noise

    def aenn(self, x, y, k=5, scale=False):

        if (scale):
            scaledData = self._scale(x)

        for i in range(k):
            self.removedIndexes = self.removedIndexes.union(
                self.enn(x=scaledData if scale else x, y=y, k=i + 1))

        setComplement = list({n for n in range(len(y))} - self.removedIndexes)
        self.cleanx = x[setComplement]
        self.cleany = y[setComplement]

        return self.removedIndexes


from sklearn import datasets
if __name__ == '__main__':
    f = AENN()

    iris = datasets.load_iris()

    print(f.aenn(x=iris.data, y=iris.target, scale=False))

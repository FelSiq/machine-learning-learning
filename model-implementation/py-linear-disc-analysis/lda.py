"""
Simple implementation of (Fisher's) Linear Discriminant Analysis.
Thanks to: https://www.python-course.eu/linear_discriminant_analysis.php
"""
import numpy as np


class LDA:
    def fit(self, X, y):
        """Fit dataset into LDA model."""
        self.X = np.array(X)
        self.y = np.array(y)

        self.classes = np.unique(y)

    def _scatter_within(self):
        num_attr = self.y.size

        sw = np.zeros((num_attr, num_attr))

        for cls in self.classes:
            class_inst = self.X[self.y == cls, :]

            class_mean = class_inst.mean(axis=0)

            class_center = class_inst - class_mean

            sw += np.matmul(class_center.T, class_center)

        return sw

    def predict(self):
        """."""
        sw = self._scatter_within()
        return sw


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()

    model = LDA()
    model.fit(iris.data, iris.target)
    model.fit([[4.1], [2.08], [0.604]], [1, 1, 1])
    ans = model.predict()

    print(ans / iris.data.shape[0])
    print(np.cov(iris.data, rowvar=False))

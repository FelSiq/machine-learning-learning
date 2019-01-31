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

        self.classes, self.cls_freqs = np.unique(y, return_counts=True)

    def _scatter_within(self):
        """."""
        sw = np.array([
            cls_freq * np.cov(self.X[self.y == cls, :], rowvar=False)
            for cls, cls_freq in zip(self.classes, self.cls_freqs)
        ]).sum(axis=0)

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
    ans = model.predict()

    print(ans)

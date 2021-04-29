"""
Simple implementation of (Fisher's) Linear Discriminant Analysis.
Thanks to: https://www.python-course.eu/linear_discriminant_analysis.php

The L. D. Matrix is a transformation matrix which best separates
the instances of different classes in data projection.
"""
import sklearn.base
import numpy as np
import scipy.linalg


class LDA(sklearn.base.TransformerMixin):
    def __init__(self, max_dim: int = -1):
        self.max_dim = int(max_dim)
        self.classes = np.empty(0)
        self.cls_freqs = np.empty(0)
        self.eig_vals = np.empty(0)
        self.transf_mat = np.empty(0)

    def _scatter_within(self, X: np.ndarray, y: np.ndarray):
        """This measure describes how scattered are each class."""
        scatter_within = np.array(
            [
                np.cov(X[y == cls, :], ddof=cls_freq - 1, rowvar=False)
                for cls, cls_freq in zip(self.classes, self.cls_freqs)
            ]
        ).sum(axis=0)

        return scatter_within

    def _scatter_between(self, X: np.ndarray, y: np.ndarray):
        """This measure describes the separation between different classes."""
        class_means = np.array([X[y == cls, :].mean(axis=0) for cls in self.classes])

        total_mean = X.mean(axis=0)

        scatter_factor = class_means - total_mean

        scatter_between = np.array(
            [
                freq * np.outer(sf, sf)
                for freq, sf in zip(self.cls_freqs, scatter_factor)
            ]
        ).sum(axis=0)

        return scatter_between

    def _get_eig(self, sw, sb):
        """Get eigenval/vec from (ScatterWithin)^(-1)*(ScatterBetween) mat."""
        sw_inv = np.eye(sw.shape[0])
        sw_inv = scipy.linalg.solve(
            sw, sw_inv, assume_a="pos", overwrite_b=True, check_finite=False
        )
        return np.linalg.eigh(np.matmul(sw_inv, sb))

    def _project(self, eig):
        """Get the K (``num_dim``) most expressive eigenvalues/vectors."""
        eig_vals, eig_vecs = eig

        eig_vals, eig_vecs = zip(
            *sorted(zip(eig_vals, eig_vecs), key=lambda item: item[0], reverse=True)[
                : self.max_dim
            ]
        )

        return eig_vals, eig_vecs

    def fit(self, X, y):
        """Fit dataset into LDA model."""
        X = np.asfarray(X)
        y = np.asarray(y)

        _, num_col = X.shape

        self.classes, self.cls_freqs = np.unique(y, return_counts=True)

        sw = self._scatter_within(X, y)
        sb = self._scatter_between(X, y)

        self.max_dim = self.max_dim if self.max_dim >= 1 else num_col
        self.max_dim = min(self.max_dim, self.classes.size - 1, num_col)

        eig = self._get_eig(sw, sb)

        eig_vals, eig_vecs = self._project(eig)

        self.eig_vals = np.array(eig_vals)
        self.transf_mat = np.concatenate(eig_vecs).reshape(num_col, self.max_dim)

        return self

    def transform(self, X, y=None):
        """Create transf. matrix which best separates the fitted data proj."""
        return np.dot(X, self.transf_mat)

    def wilks_lambda(self):
        """Compute Wilks' Lambda measure using eigenvalues of L. D. matrix."""
        return np.prod(1.0 / (1.0 + self.eig_vals))

    def canonical_corr(self):
        """Calculate canonical correlation values from L. D. matrix."""
        return (self.eig_vals / (1.0 + self.eig_vals)) ** 0.5


if __name__ == "__main__":
    from sklearn import datasets

    X, y = datasets.load_iris(return_X_y=True)

    model = LDA()
    ans = model.fit_transform(X, y)

    print("Transformation Matrix:", model.transf_mat, sep="\n", end="\n\n")
    print("Eigenvalues of L. D. matrix:", model.eig_vals, end="\n\n")
    print("Canonical Correlation:", model.canonical_corr(), end="\n\n")
    print("Wilks' Lambda:", model.wilks_lambda())

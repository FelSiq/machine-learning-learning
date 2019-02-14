"""
Simple implementation of (Fisher's) Linear Discriminant Analysis.
Thanks to: https://www.python-course.eu/linear_discriminant_analysis.php

The L. D. Matrix is a transformation matrix which best separates
the instances of different classes in data projection.
"""
import numpy as np


class LDA:
    def fit(self, X, y):
        """Fit dataset into LDA model."""
        self.X = np.array(X)
        self.y = np.array(y)

        self.classes, self.cls_freqs = np.unique(y, return_counts=True)

    def _scatter_within(self):
        """This measure describes how scattered are each class."""
        scatter_within = np.array([
            (cls_freq - 1) * np.cov(self.X[self.y == cls, :], rowvar=False)
            for cls, cls_freq in zip(self.classes, self.cls_freqs)
        ]).sum(axis=0)

        return scatter_within

    def _scatter_between(self):
        """This measure describes the separation between different classes."""
        class_means = np.array(
            [self.X[self.y == cls, :].mean(axis=0) for cls in self.classes])

        total_mean = self.X.mean(axis=0)

        scatter_factor = class_means - total_mean

        scatter_between = np.array([
            freq * np.outer(sf, sf)
            for freq, sf in zip(self.cls_freqs, scatter_factor)
        ]).sum(axis=0)

        return scatter_between

    def _get_eig(self, sw, sb):
        """Get eigenval/vec from (ScatterWithin)^(-1)*(ScatterBetween) mat."""
        sw_inv = np.linalg.inv(sw)

        return np.linalg.eig(np.matmul(sw_inv, sb))

    def _project(self, eig, num_dim):
        """Get the K (``num_dim``) most expressive eigenvalues/vectors."""
        eig_vals, eig_vecs = eig

        eig_vals, eig_vecs = zip(
            *sorted(
                zip(eig_vals, eig_vecs),
                key=lambda item: item[0],
                reverse=True)[:num_dim])

        return eig_vals, eig_vecs

    def predict(self, max_dim=2):
        """Create transf. matrix which best separates the fitted data proj."""
        sw = self._scatter_within()
        sb = self._scatter_between()

        max_dim = min(max_dim, self.classes.size-1)

        eig = self._get_eig(sw, sb)

        eig_vals, eig_vecs = self._project(eig, num_dim=max_dim)

        _, num_col = self.X.shape

        self.eig_vals = np.array(eig_vals)
        self.transf_mat = np.concatenate(eig_vecs).reshape(num_col, max_dim)

        self.transf_mat = self.transf_mat.real

        return self.transf_mat

    def wilks_lambda(self):
        """Compute Wilks' Lambda measure using eigenvalues of L. D. matrix."""
        return np.prod(1.0 / (1.0 + self.eig_vals))

    def canonical_corr(self):
        """Calculate canonical correlation values from L. D. matrix."""
        return (self.eig_vals / (1.0 + self.eig_vals))**0.5


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()

    model = LDA()
    model.fit(iris.data, iris.target)
    ans = model.predict(max_dim=2)

    print("Transformation Matrix:", ans, sep="\n", end="\n\n")
    print("Eigenvalues of L. D. matrix:", model.eig_vals, end="\n\n")
    print("Canonical Correlation:", model.canonical_corr(), end="\n\n")
    print("Wilks' Lambda:", model.wilks_lambda())

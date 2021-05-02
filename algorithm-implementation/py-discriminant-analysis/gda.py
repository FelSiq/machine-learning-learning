"""Simple implementation of Gaussian Discriminant Analysis Classifier algorithm.

The Gaussian discriminant analysis can refer to either Linear Discriminent Analysis
(LDA) or Quadratic Discriminant Analysis (QDA).

The only difference implementation- wise is a single line in the code:
- In LDA, the covariance matrix for every class is assumed to be the same.
- In QDA, the covaraince matrices for each class is not assumed to be the same.

Geometrically, this has a deeper interpretation:
- In LDA the boundary is constitued by degree 1 polynomials, and is equivalent to a
  Softmax Classification with degree 1 polynomials to compute the class logits.
- In QDA, the boundary is constitued by degree 2 polynomials, and is equivalent to a
  Softmax Classification with degree 2 polynomials to compute the class logits.

Note that, even to both LDA and QDA imply the use of Softmax Classification, the
converse is not true since Softmax Classification does not assumes that the independent
attributes (X) comes from a normal distribution and therefore can potentially
discriminate classes with a broader class of hypotheses available.
"""
import typing as t

import numpy as np
import scipy.stats


class GDA:
    """Simple Gaussian Discriminant Analysis Classifier model.

    This algorithm chooses between Linear Discriminant Analysis (LDA) and
    Quadratic Discriminant Analysis (QDA) depending on its argument "quadratic".
    """

    def __init__(self, quadratic: bool = False):
        self._logpriors = np.empty(0)
        self._classes = np.empty(0)
        self._cls_freqs = np.empty(0)
        self.quadratic = quadratic

        self._dists = []  # type: t.List[scipy.stats.multivariate_normal]

    @staticmethod
    def _check_X(X: np.ndarray) -> None:
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

    @staticmethod
    def _check_y(y: np.ndarray) -> None:
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1

    def _calc_gaussian_dists(self, X: np.ndarray, y: np.ndarray) -> None:
        self._dists = []

        means = []
        covs = []

        if not self.quadratic:
            n, m = X.shape
            shared_cov = np.zeros((m, m), dtype=float)

        for cls, cls_ind in enumerate(self._classes):
            X_slice = X[self._classes[cls_ind] == y, :]

            cls_mean = np.mean(X_slice, axis=0)

            cls_cov = np.cov(
                X_slice,
                rowvar=False,
                ddof=0 if self.quadratic else X_slice.shape[0] - 1,
            )

            if not self.quadratic:
                shared_cov += cls_cov / n
                cls_cov = shared_cov

            means.append(cls_mean)
            covs.append(cls_cov)

        for mean, cov in zip(means, covs):
            dist = scipy.stats.multivariate_normal(
                mean=mean, cov=cov, allow_singular=True
            )
            self._dists.append(dist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GDA":
        self._check_X(X)
        self._check_y(y)
        assert X.shape[0] == y.size

        self._classes, self._cls_freqs = np.unique(y, return_counts=True)
        self._logpriors = np.log(self._cls_freqs.astype(float) / y.size)
        self._calc_gaussian_dists(X, y)

        return self

    def predict(self, X: np.ndarray) -> t.List[t.Any]:
        self._check_X(X)

        preds = np.full(X.shape[0], fill_value=-1, dtype=int)
        best_posterior_logprobs = np.full(len(preds), fill_value=-np.inf)

        for cls_ind in range(len(self._classes)):
            likelihoods = self._dists[cls_ind].logpdf(X)
            cls_posterior_logprobs = likelihoods + self._logpriors[cls_ind]

            update_inds = best_posterior_logprobs < cls_posterior_logprobs

            preds[update_inds] = cls_ind
            best_posterior_logprobs[update_inds] = cls_posterior_logprobs[update_inds]

        preds = self._classes[preds]

        return preds


def _test_a():
    import sklearn.metrics
    import sklearn.model_selection
    import sklearn.preprocessing
    import sklearn.datasets
    import sklearn.discriminant_analysis

    quadratic = True

    X, y = sklearn.datasets.load_iris(return_X_y=True)

    n_splits = 10
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=16
    )

    avg_acc = avg_acc_ref = 0.0

    if quadratic:
        ref = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()

    else:
        ref = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

    model = GDA(quadratic=quadratic)

    for train_inds, test_inds in splitter.split(X, y):
        X_train, X_test = X[train_inds, ...], X[test_inds, ...]
        y_train, y_test = y[train_inds], y[test_inds]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        avg_acc += sklearn.metrics.accuracy_score(preds, y_test)

        ref.fit(X_train, y_train)
        preds_ref = ref.predict(X_test)
        avg_acc_ref += sklearn.metrics.accuracy_score(preds_ref, y_test)

    avg_acc /= n_splits
    avg_acc_ref /= n_splits

    print("10-fold avg accuracy (mine)     :", avg_acc)
    print("10-fold avg accuracy (reference):", avg_acc_ref)


def _test_b():
    import matplotlib.pyplot as plt
    import sklearn.discriminant_analysis
    import sklearn.preprocessing
    import scipy.stats
    import torch
    import torch.nn as nn

    class SoftmaxClassification(nn.Module):
        def __init__(self, degree: int = 1, n_input: int = 2, n_class: int = 3):
            super(SoftmaxClassification, self).__init__()
            assert int(degree) >= 1
            self.degree = int(degree)
            self.n_class = int(n_class)

            self.poly = sklearn.preprocessing.PolynomialFeatures(
                degree=self.degree, include_bias=False
            )
            self.poly.fit(np.empty((1, n_input)))

            self.weights = nn.Linear(
                self.poly.n_output_features_, self.n_class, bias=True
            )

        def fit(self, X, y):
            X = self.poly.transform(X)
            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)

            criterion = nn.CrossEntropyLoss()
            optim = torch.optim.RMSprop(self.parameters(), lr=0.1)

            for i in np.arange(200):
                optim.zero_grad()
                y_preds = self(X)
                loss = criterion(y_preds, y)
                loss.backward()
                optim.step()

            return self

        def forward(self, X):
            logits = self.weights(X)

            if self.training:
                return logits

            return logits.argmax(dim=-1)

        def predict(self, X):
            if not self.training:
                X = self.poly.transform(X)
                X = torch.tensor(X, dtype=torch.float)

            return self.forward(X).detach().numpy().ravel()

    def gen_data(mean, cov, cls_ind, n=75):
        dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        return dist.rvs(n), np.full(n, fill_value=cls_ind, dtype=int)

    def shuffle(X, y):
        inds = np.arange(y.size)
        np.random.shuffle(inds)

        X[...] = X[inds, :]
        y[...] = y[inds]

    np.random.seed(8)

    X1, y1 = gen_data([0.0, 2.0], [[1, 0.25], [0.25, 1]], 0)
    X2, y2 = gen_data([-2.0, 2.0], [[0.5, -0.125], [-0.125, 0.5]], 1)
    X3, y3 = gen_data([1.0, 0.2], [[0.3, -0.225], [-0.225, 0.3]], 2)

    X, y = np.vstack((X1, X2, X3)), np.concatenate((y1, y2, y3))
    shuffle(X, y)

    mapper_plot = {0: (1.0, 0.8, 0.8), 1: (0.8, 0.8, 1.0), 2: (1.0, 0.95, 0.7)}
    mapper_data = {0: (0.9, 0.07, 0.27), 1: (0.0, 0.47, 1.0), 2: (1.0, 0.8, 0.0)}

    def map_color(y, mapper_color):
        return list(map(mapper_color.get, y))

    data_colors = map_color(y, mapper_data)

    models = (
        ("LDA", GDA()),
        ("sklearn LDA", sklearn.discriminant_analysis.LinearDiscriminantAnalysis()),
        ("softmax degree 1", SoftmaxClassification(degree=1)),
        ("QDA", GDA(quadratic=True)),
        ("sklearn QDA", sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()),
        ("softmax degree 2", SoftmaxClassification(degree=2)),
    )

    min_, max_ = np.quantile(X, (0, 1), axis=0)
    t = np.linspace(min_, max_, 150)
    plot_X, plot_Y = np.meshgrid(t[:, 0], t[:, 1], copy=False)
    plot_data = np.column_stack((plot_X.ravel(), plot_Y.ravel()))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

    def find_boundaries(plot_preds):
        plot_preds = plot_preds.reshape(t.shape[0], t.shape[0])
        plot_preds = plot_preds.astype(int, copy=False)

        boundary_horiz = plot_preds[:, 1:] != plot_preds[:, :-1]
        boundary_vert = plot_preds[1:, :] != plot_preds[:-1, :]

        boundary_horiz = np.pad(boundary_horiz, ((0, 0), (1, 0)))
        boundary_vert = np.pad(boundary_vert, ((1, 0), (0, 0)))

        boundary = np.logical_or(boundary_horiz, boundary_vert)

        return set(np.flatnonzero(boundary.ravel()))

    for i, (model_name, model) in enumerate(models):
        model.fit(X, y)

        try:
            model.eval()

        except AttributeError:
            pass

        plot_preds = model.predict(plot_data)
        plot_colors = map_color(plot_preds, mapper_plot)

        boundary_inds = find_boundaries(plot_preds)

        for j in boundary_inds:
            plot_colors[j] = (0.0, 0.0, 0.0)

        ax = axes[i // 3][i % 3]
        ax.set_title(model_name)
        ax.scatter(*plot_data.T, s=4, c=plot_colors)
        ax.scatter(*X.T, s=4, c=data_colors)

    plt.show()


if __name__ == "__main__":
    # _test_a()
    _test_b()

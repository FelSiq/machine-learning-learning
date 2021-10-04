"""Non-negative matrix factorization using Gradient Descent."""
import typing as t

import numpy as np
import sklearn.base


class NMF(sklearn.base.TransformerMixin):
    def __init__(
        self,
        n_topics: int,
        max_iter: int = 256,
        learning_rate: float = 1.0,
        random_state: t.Optional[int] = None,
    ):
        assert int(max_iter) > 0
        assert int(n_topics) > 0
        assert float(learning_rate) > 0

        self.n_topics = int(n_topics)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)

        self.random_state = random_state

        self.H = np.empty((0, 0), dtype=float)

    def _fit_transform(self, X, update_h: bool = True):
        X = np.asfarray(X)

        n, m = X.shape

        X_scaled_mean = float(np.sqrt(X.mean() / self.n_topics))

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if update_h or self.H.size == 0:
            W = np.random.random((n, self.n_topics)) * X_scaled_mean
            self.H = np.random.random((self.n_topics, m)) * X_scaled_mean

        else:
            W = np.full((n, self.n_topics), fill_value=X_scaled_mean, dtype=float)

        for i in np.arange(self.max_iter):
            X_approx = W @ self.H
            W_update = (X @ self.H.T) / (1e-6 + X_approx @ self.H.T)

            if update_h:
                H_update = (W.T @ X) / (1e-6 + W.T @ X_approx)
                self.H *= H_update

            W *= W_update

            if i % 100 == 0:
                error = float(np.linalg.norm(X - W @ self.H, ord=2, axis=None))
                print(
                    f"({'Fit' if update_h else 'Transform'}) "
                    f"Reconstruction error: {error:.4f}"
                )

        return W

    def fit(self, X, y=None):
        self._fit_transform(X=X, update_h=True)
        return self

    def transform(self, X):
        # Fix H, create new W
        W = self._fit_transform(X=X, update_h=False)
        return W


def _test():
    import sklearn.decomposition
    import sklearn.datasets
    import matplotlib.pyplot as plt

    X, _ = sklearn.datasets.load_iris(return_X_y=True)
    np.random.seed(32)
    np.random.shuffle(X)

    n_topics = 2

    ref = sklearn.decomposition.NMF(
        n_components=n_topics,
        max_iter=1024,
        random_state=16,
        init="random",
        solver="mu",
    )
    model = NMF(n_topics=n_topics, max_iter=1024, random_state=16)

    res_ref = ref.fit_transform(X)
    res_model = model.fit_transform(X)

    assert np.all(res_model >= 0.0)

    m = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)
    ax1.imshow(res_ref[:m])
    ax2.imshow(res_model[:m])
    plt.show()


if __name__ == "__main__":
    _test()

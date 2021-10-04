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

    @staticmethod
    def _clip_grad_norm(X, max_grad_norm: float = 1.0):
        clip_coef = max_grad_norm / (1e-6 + float(np.linalg.norm(X, ord=2, axis=None)))

        if clip_coef >= 1.0:
            return X

        X *= clip_coef

        return X

    def _fit_transform(self, X, update_h: bool = True):
        X = np.asfarray(X)

        n, m = X.shape

        X_scaled_mean = float(np.sqrt(X.mean() / self.n_topics))

        if self.random_state is not None:
            np.random.seed(self.random_state)

        W = np.random.random((n, self.n_topics)) * X_scaled_mean

        if update_h:
            self.H = np.random.random((self.n_topics, m)) * X_scaled_mean

        for i in np.arange(self.max_iter):
            # Update W given H
            base_grad = (W @ self.H - X) / n
            W_grad = base_grad @ self.H.T
            self._clip_grad_norm(W_grad)
            W -= self.learning_rate * W_grad
            np.maximum(W, 0.0, out=W)

            if update_h:
                # Update H given W
                H_grad = W.T @ base_grad
                self._clip_grad_norm(H_grad)
                self.H -= self.learning_rate * H_grad
                np.maximum(self.H, 0.0, out=self.H)

            if i % 100 == 0:
                print(
                    f"Reconstruction error: {float(np.linalg.norm(base_grad, ord=2)):.4f}"
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

    X, _ = sklearn.datasets.load_breast_cancer(return_X_y=True)

    n_topics = 3

    ref = sklearn.decomposition.NMF(
        n_components=n_topics, max_iter=512, random_state=16, init="random"
    )
    model = NMF(n_topics=n_topics, max_iter=512, random_state=16)

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

import numpy as np
import sklearn.base
import functools

import dist_grad


class ICA(sklearn.base.TransformerMixin):
    def __init__(
        self,
        n_components: int,
        learning_rate: float = 1e-3,
        max_epochs: int = 512,
        batch_size: int = 32,
        source_dist: str = "sigmoid",
        *args,
        **kwargs,
    ):
        assert int(n_components) >= 2
        assert int(max_epochs) > 0
        assert int(batch_size) > 0
        assert float(learning_rate) > 0.0
        assert source_dist in {"sigmoid", "laplace"}

        self.n_components = int(n_components)
        self.source_dist = source_dist
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)

        source_dist = functools.partial(
            dist_grad.get_dist_grad_fun(self.source_dist), *args, **kwargs
        )
        self._source_dist_grad_fun = source_dist

        self.unmixing_matrix = np.empty(0)

    def _run_epoch(self, X, batch_inds):
        n, _ = X.shape

        for start in np.arange(0, n, self.batch_size):
            end = start + self.batch_size
            X_batch = X[:, batch_inds[start:end]]

            source_dist_grad = self._source_dist_grad_fun(X_batch)

            grad = (
                np.dot(source_dist_grad, X_batch.T)
                + np.linalg.inv(self.unmixing_matrix).T
            )

            update = self.learning_rate * grad / self.batch_size
            self.unmixing_matrix -= update

    def fit(self, X, y=None):
        X = np.asfarray(X)

        assert X.ndim == 2

        d, n = X.shape

        self.unmixing_matrix = np.random.randn(d, d)
        batch_inds = np.arange(n)

        for i in np.arange(1, 1 + n):
            np.random.shuffle(batch_inds)
            self._run_epoch(X, batch_inds)

        return self

    def transform(self, X, y=None):
        return self.unmixing_matrix @ X


def _test():
    import matplotlib.pyplot as plt

    np.random.seed(16)

    t = np.linspace(0, 10, 1000)
    signal_1 = np.sin(2.0 * np.pi * 1.5 * t) + 0.25 * np.random.randn(t.size)
    signal_2 = 2 * np.cos(2.0 * np.pi * 0.33 * t) + 1.5 * np.random.randn(t.size)

    mix_1 = 0.7 * signal_1 + 0.3 * signal_2
    mix_2 = 0.4 * signal_1 + 0.6 * signal_2

    mixes = np.vstack((mix_1, mix_2))

    print(mixes.shape)

    model = ICA(n_components=2, source_dist="sigmoid")
    sources = model.fit_transform(mixes)

    recovered_2, recovered_1 = sources

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 10))

    color_1 = np.array([255, 140, 0], dtype=float) / 255.0
    color_2 = np.array([0, 140, 255], dtype=float) / 255.0

    ax1.plot(signal_1, color=tuple(color_1))
    ax2.plot(signal_2, color=tuple(color_2))

    ax3.plot(mix_1, color=tuple(0.7 * color_1 + 0.3 * color_2), label="original")
    ax3.plot(-1 * recovered_1, color=(1, 0, 0), label="recovered")
    ax3.legend()

    ax4.plot(mix_2, color=tuple(0.4 * color_1 + 0.6 * color_2), label="original")
    ax4.plot(-1 * recovered_2, color=(1, 0, 0), label="recovered")
    ax4.legend()

    plt.show()


if __name__ == "__main__":
    _test()

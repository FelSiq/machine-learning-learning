import numpy as np
import sklearn.base
import functools

import dist_grad


class ICA(sklearn.base.TransformerMixin):
    def __init__(
        self,
        n_components: int,
        learning_rate: float = 1e-4,
        max_epochs: int = 512,
        batch_size: int = 32,
        source_dist: str = "sigmoid",
        threshold: float = 1e-5,
        *args,
        **kwargs,
    ):
        assert int(n_components) >= 2
        assert int(max_epochs) > 0
        assert int(batch_size) > 0
        assert float(learning_rate) > 0.0
        assert source_dist in {"sigmoid", "laplace"}
        assert float(threshold) >= 0.0

        self.n_components = int(n_components)
        self.source_dist = source_dist
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.threshold = float(threshold)

        source_dist = functools.partial(
            dist_grad.get_dist_grad_fun(self.source_dist), *args, **kwargs
        )
        self._source_dist_grad_fun = source_dist

        self.unmixing_matrix = np.empty(0)

    def _run_epoch(self, X, batch_inds):
        n, _ = X.shape
        max_diff = -np.inf

        for start in np.arange(0, n, self.batch_size):
            end = start + self.batch_size
            X_batch = X[batch_inds[start:end], :]

            transform = self.transform(X_batch)

            source_dist_grad = self._source_dist_grad_fun(transform)

            grad = np.dot(
                X_batch.T, -source_dist_grad
            ) - self.batch_size * np.linalg.inv(self.unmixing_matrix.T)

            update = self.learning_rate * grad / self.batch_size
            self.unmixing_matrix -= update

            if self.threshold <= 0.0:
                max_diff = np.inf
                continue

            max_diff = max(max_diff, np.linalg.norm(update, ord=np.inf))

        return max_diff

    def fit(self, X, y=None):
        X = np.asfarray(X)

        assert X.ndim == 2

        n, d = X.shape

        assert n > d

        self.unmixing_matrix = np.random.randn(d, d)
        batch_inds = np.arange(n)

        for i in np.arange(1, 1 + self.max_epochs):
            np.random.shuffle(batch_inds)
            max_diff = self._run_epoch(X, batch_inds)

            if max_diff < self.threshold:
                print(f"Early stopping at epoch {i}.")
                break

        return self

    def transform(self, X, y=None):
        X = np.asfarray(X)
        return X @ self.unmixing_matrix


def _test():
    import matplotlib.pyplot as plt
    import sklearn.decomposition

    np.random.seed(8)

    t = np.linspace(0, 10, 200)
    signal_1 = np.sin(2.0 * np.pi * 0.5 * t) + 0.1 * np.random.randn(t.size)
    signal_2 = np.cumsum(np.random.randn(t.size))
    signal_2 = (signal_2 - np.min(signal_2)) / np.ptp(signal_2)

    signal_1 -= np.mean(signal_1)
    signal_2 -= np.mean(signal_2)

    mix_1 = 0.7 * signal_1 + 0.3 * signal_2
    mix_2 = 0.2 * signal_1 + 0.8 * signal_2

    mixes = np.column_stack((mix_1, mix_2))

    print(mixes.shape)

    model = ICA(n_components=2, source_dist="sigmoid", batch_size=1)
    ref = sklearn.decomposition.FastICA()
    sources = model.fit_transform(mixes)
    sources_ref = ref.fit_transform(mixes)

    recovered_2, recovered_1 = sources.T
    recovered_2_ref, recovered_1_ref = sources_ref.T

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 10))

    color_1 = np.array([255, 140, 0], dtype=float) / 255.0
    color_2 = np.array([0, 140, 255], dtype=float) / 255.0

    ax1.plot(recovered_1, color=(1, 0, 0), label="recovered")
    ax1.plot(
        recovered_1_ref, color=(1, 0, 0.8), label="recovered (sklearn)", linestyle="-."
    )
    ax1.plot(signal_1, color=tuple(color_1), label="original")
    ax1.set_title("Original signal 1")
    ax1.legend()

    ax2.plot(recovered_2, color=(1, 0, 0), label="recovered")
    ax2.plot(
        recovered_2_ref, color=(1, 0, 0.8), label="recovered (sklearn)", linestyle="-."
    )
    ax2.plot(signal_2, color=tuple(color_2), label="original")
    ax2.set_title("Original signal 2")
    ax2.legend()

    ax3.plot(mix_1, color=tuple(0.7 * color_1 + 0.3 * color_2))
    ax3.set_title("Mixture 1")

    ax4.plot(mix_2, color=tuple(0.2 * color_1 + 0.8 * color_2))
    ax4.set_title("Mixture 2")

    plt.show()


if __name__ == "__main__":
    _test()

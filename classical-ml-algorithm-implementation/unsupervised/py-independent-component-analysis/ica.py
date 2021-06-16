import numpy as np
import sklearn.base
import functools

import dist_grad


class ICA(sklearn.base.TransformerMixin):
    def __init__(
        self,
        learning_rate: float = 1e-2,
        max_epochs: int = 512,
        batch_size: int = 32,
        source_dist: str = "laplace",
        threshold: float = 1e-3,
        *args,
        **kwargs,
    ):
        assert int(max_epochs) > 0
        assert int(batch_size) > 0
        assert float(learning_rate) > 0.0
        assert source_dist in {"sigmoid", "laplace"}
        assert float(threshold) >= 0.0

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
        self._training = False

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

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        X = (X - self.mean) / self.std

        self.unmixing_matrix = np.random.randn(d, d)
        batch_inds = np.arange(n)
        self._training = True

        for i in np.arange(1, 1 + self.max_epochs):
            np.random.shuffle(batch_inds)
            max_diff = self._run_epoch(X, batch_inds)

            if max_diff < self.threshold:
                print(f"Early stopping at epoch {i}.")
                break

        self._training = False

        return self

    def transform(self, X, y=None):
        if not self._training:
            X = np.asfarray(X) - self.mean

        transf = X @ self.unmixing_matrix

        if not self._training:
            transf += self.mean

        return transf


def _test():
    import matplotlib.pyplot as plt
    import sklearn.decomposition
    import pandas as pd

    np.random.seed(16)

    def prepare_mixtures():
        data = pd.read_csv("data.csv", index_col=0, squeeze=True)
        max_size = 500
        X = np.asfarray([v.split(",")[:max_size] for v in data.values])

        signals = X[[9, 2, 7], :].T
        signals = (signals - np.min(signals, axis=0)) / np.ptp(signals, axis=0)

        mix_mat = np.random.random((3, 3))
        mix_mat /= np.sum(mix_mat, axis=0)

        mixes = signals @ mix_mat

        assert mixes.shape == signals.shape

        return signals, mixes, mix_mat

    signals, mixes, mix_mat = prepare_mixtures()

    num_signals = signals.shape[1]

    model = ICA(
        learning_rate=5e-2,
        source_dist="laplace",
        batch_size=256,
        max_epochs=1024,
        threshold=1e-4,
    )
    ref = sklearn.decomposition.FastICA(max_iter=1024, whiten=False, random_state=16)

    reconstituted = model.fit_transform(mixes)
    reconstituted_ref = ref.fit_transform(mixes)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharey=True, sharex=True)

    colors = (
        np.array(
            [
                [255, 140, 0],
                [0, 140, 255],
                [70, 250, 70],
            ],
            dtype=float,
        )
        / 255.0
    )

    colors_mix = colors @ mix_mat

    # NOTE: ICA result as ambiguous to mirroring and signal ordering,
    # so we may need to adjust both.
    reconstituted = reconstituted[:, [0, 2, 1]]
    mean = np.mean(reconstituted, axis=0)
    reconstituted -= mean
    reconstituted[:, [1]] *= -1.0
    reconstituted += mean

    reconstituted_ref = reconstituted_ref[:, [1, 2, 0]]

    for i in np.arange(num_signals):
        ax_mix, ax_sig = axes[i]

        ax_mix.plot(mixes[:, i], color=colors_mix[i])
        ax_mix.set_title(f"Mixture {i + 1}")

        ax_sig.plot(reconstituted_ref[:, i], label="reference (sklearn)", c="black")
        ax_sig.plot(reconstituted[:, i], label="reconstituted", c="purple")
        ax_sig.plot(signals[:, i], label="original", c=colors[i])
        ax_sig.legend()
        ax_sig.set_title(f"Signal {i + 1}")

    plt.show()


if __name__ == "__main__":
    _test()

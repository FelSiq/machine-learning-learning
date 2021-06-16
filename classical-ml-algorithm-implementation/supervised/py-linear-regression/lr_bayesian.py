import typing as t

import numpy as np
import scipy.stats


class LinRegBayesian:
    def __init__(
        self,
        coeffs_prior_var: float,
        label_noise_var: t.Optional[float] = None,
        add_intercept: bool = True,
        quantiles: t.Optional[t.Sequence[float]] = None,
    ):
        assert label_noise_var is None or float(label_noise_var) > 0.0
        assert float(coeffs_prior_var) > 0.0

        self.add_intercept = add_intercept

        self.coeffs_prior_var = float(coeffs_prior_var)
        self.label_noise_var = label_noise_var

        if quantiles is None:
            quantiles = np.empty(0)

        self.quantiles = np.asfarray(quantiles)
        self.dist_mean = np.empty(0)
        self.dist_cov = np.empty(0)

        assert np.all(np.logical_and(0.0 <= self.quantiles, 1.0 >= self.quantiles))

    def fit(self, X, y):
        X = np.asfarray(X)
        y = np.asfarray(y)

        assert X.ndim == 2
        assert X.shape[0] == y.size

        if self.label_noise_var is None:
            self.label_noise_var = np.var(y, ddof=1)

        n, m = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        A = (
            X.T @ X / self.label_noise_var
            + np.eye(m + self.add_intercept) / self.coeffs_prior_var
        )

        self.dist_cov = np.linalg.inv(A)
        self.dist_mean = self.dist_cov @ X.T @ y / self.label_noise_var

        return self

    def predict(self, X):
        n, _ = X.shape

        if self.add_intercept:
            X = np.column_stack((np.ones(n, dtype=float), X))

        preds = np.zeros((n, 1 + self.quantiles.size), dtype=float)

        for i in np.arange(n):
            cur_inst = X[i, :]

            mean = cur_inst.T @ self.dist_mean
            std = np.sqrt(cur_inst.T @ self.dist_cov @ cur_inst + self.label_noise_var)

            cur_dist = scipy.stats.norm(loc=mean, scale=std)

            cur_pred = cur_dist.mean()
            preds[i, 0] = cur_pred

            if self.quantiles.size:
                cur_quantiles = cur_dist.ppf(self.quantiles)
                preds[i, 1:] = cur_quantiles

        return np.squeeze(preds)


def _test():
    import test
    import matplotlib.pyplot as plt
    import sklearn.model_selection

    import lr_analytical

    test.test(model=LinRegBayesian(coeffs_prior_var=1000.0))

    model = LinRegBayesian(coeffs_prior_var=100.0, quantiles=(0.05, 0.95, 0.15, 0.85))

    X = np.random.random((512, 1))
    y = np.dot(X, [20]) - 5 + 2 * np.random.randn(X.shape[0])
    X_train, X_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        X, y, test_size=0.05, shuffle=True
    )

    model.fit(X_train, y_train)
    y_preds = model.predict(X_eval)
    y_preds[:, 1:] = np.abs(y_preds[:, 1:] - y_preds[:, 0, np.newaxis])

    ref = lr_analytical.LinReg().fit(X_train, y_train)
    y_ref = ref.predict(X_eval)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.scatter(X_train, y_train, label="train")
    ax.scatter(X_eval, y_eval, label="eval")
    ax.errorbar(X_eval, y_preds[:, 0], yerr=y_preds[:, [1, 2]].T, label="predict 90%")
    ax.errorbar(X_eval, y_preds[:, 0], yerr=y_preds[:, [3, 4]].T, label="predict 70%")
    ax.plot(X_eval, y_ref, label="analytical lin reg", linestyle="-.", color="purple")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    _test()

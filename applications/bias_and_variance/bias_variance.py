import typing as t

import numpy as np
import sklearn.preprocessing


def decompose_mse(
    degrees: t.Sequence[int],
    true_std: float,
    true_underlying_func: t.Callable[[t.Sequence[float]], t.Sequence[float]],
    n: int = 2000,
    train_size: int = 50,
    xlim=(-10, 10),
):
    if np.isscalar(degrees):
        degrees = [degrees]

    designers = [
        sklearn.preprocessing.PolynomialFeatures(degree=degree) for degree in degrees
    ]

    model = sklearn.linear_model.LinearRegression(fit_intercept=False, copy_X=False)

    xmin, xmax = xlim
    true = np.empty((n, 1))
    preds = np.empty((n, len(degrees)))

    X_test = (xmax - xmin) * np.random.random(1) + xmin
    X_test = X_test.reshape(-1, 1)

    X_test_designs = [designer.fit_transform(X_test) for designer in designers]
    coeffs = [np.zeros(1 + degree, dtype=float) for degree in degrees]

    for i in np.arange(n):
        X_train = (xmax - xmin) * np.random.random(train_size) + xmin
        y_train = true_underlying_func(X_train)
        y_test = true_underlying_func(X_test.ravel())

        X_train = X_train.reshape(-1, 1)

        true[i] = y_test

        for j in range(len(degrees)):
            X_train_design = designers[j].transform(X_train)

            model.fit(X_train_design, y_train)
            y_preds = model.predict(X_test_designs[j])

            coeffs[j] += model.coef_
            preds[i, j] = y_preds

    direct_mse = np.mean(np.square(preds - true), axis=0)

    irredutible_err = true_std ** 2
    var = np.var(preds, axis=0)
    bias = np.mean(preds - true, axis=0)

    coeffs = [c / n for c in coeffs]

    return direct_mse, irredutible_err, var, bias, coeffs


def _test():
    import matplotlib.pyplot as plt
    import sklearn.linear_model

    true_std = 8
    train_size = 200
    degree = [1, 2, 3, 9]
    np.random.seed(32)

    def true_underlying_func(x, noise=True):
        return (
            0.1 * x ** 3
            - 1 * x ** 2
            + 5 * x
            + 20
            + int(noise) * true_std * np.random.randn(x.size)
        )

    direct_mse, irredutible_err, var, bias, coeffs = decompose_mse(
        degree,
        true_std,
        true_underlying_func,
        n=500,
        train_size=train_size,
    )

    mse = irredutible_err + var + np.square(bias)

    X = np.linspace(-25, 25, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f"Linear regression - Var[$\epsilon$]: {irredutible_err:.2f} (irredutible error)"
    )

    for i, c in enumerate(coeffs):
        X_design = X.reshape(-1, 1) ** list(range(len(c)))
        preds = np.sum(X_design * c, axis=1)
        true_y = true_underlying_func(X, False)

        ax = axes[i // 2][i % 2]
        ax.set_title(
            f"Degree: {len(c) - 1} - MSE: (dec) {mse[i]:.2f}, (direct) {direct_mse[i]:.2f}\n"
            f"$Bias^{{{2}}}(f_{{{train_size}}})$: {bias[i] ** 2:.2f}, "
            f"$Var[f_{{{train_size}}}]$: {var[i]:.2f}"
        )
        ax.plot(X, true_y, label="true function")
        ax.plot(X, preds, label="mean fit pred", color="orange")
        ax.legend()

    plt.show()


if __name__ == "__main__":
    _test()

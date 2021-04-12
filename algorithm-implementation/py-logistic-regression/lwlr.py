import typing as t

import numpy as np
import scipy.spatial.distance
import matplotlib
import matplotlib.pyplot as plt


def rel_err(err, prev_err):
    return (prev_err - err) / abs(1e-6 + prev_err)


def _add_bias(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return np.hstack((np.ones((X.shape[0], 1)), X))


def calc_weights(X, X_train, tau):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    weights = np.exp(-0.5 / (tau ** 2) * scipy.spatial.distance.cdist(X, X_train))

    return weights


def predict(
    X,
    X_train,
    theta,
    tau,
    weights: t.Optional[np.ndarray] = None,
    add_bias: bool = True,
):
    if weights is None:
        weights = calc_weights(X, X_train, tau)

    if add_bias:
        X_train = _add_bias(X_train)

    return np.sum(weights * np.dot(theta, X_train.T), axis=1)


def fit(
    X,
    y,
    tau: float = 1,
    step_size: float = 1e-2,
    max_it: int = 1e8,
    max_rel_err: float = 1e-6,
    random_state: t.Optional[int] = None,
    initial_theta: t.Optional[np.ndarray] = None,
    verbose: bool = True,
):
    max_it = int(max_it)

    X = _add_bias(X)

    if initial_theta is not None:
        theta = np.asarray(initial_theta, dtype=float)

    else:
        if random_state is not None:
            np.random.seed(random_state)

        theta = np.random.randn(X.shape[1])

    it = 0
    it_to_print = 1

    weights = calc_weights(X, X, tau)

    prev_mse = 1 / (1 + max_rel_err)
    gd_path = []

    while it < max_it and not np.isnan(prev_mse):
        it += 1
        y_pred = predict(X, X, theta, tau, weights, add_bias=False)
        err = y_pred - y
        theta -= step_size * np.mean(err * X.T, axis=1)

        mse = np.mean(np.square(err))

        if verbose and it % it_to_print == 0:
            print(f"{it:<{9}} - MSE: {mse:.6f}")
            gd_path.append(np.hstack((theta, mse)))
            it_to_print = min(it_to_print + 5, 100)

        if abs(rel_err(mse, prev_mse)) <= max_rel_err:
            it = max_it

        prev_mse = mse

    print(f"Converged - MSE: {mse:.6f}")

    gd_path.append(np.hstack((theta, mse)))

    return theta, np.asarray(gd_path)


def _test():
    m = 200

    dims = 1

    X1 = np.linspace(0, 4 * np.pi, m) + 2 * np.random.randn(m)

    if dims == 2:
        X2 = (np.random.random(m) < 0.2).astype(float)

        X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        y = 0.8 * X1 - 10 * X2 + 4 * np.random.randn(m)

    else:
        X = X1
        y = 2 * np.cos(X1 * 0.25) + 0.1 * np.random.randn(m)

    X = 2 * (X - X.min(axis=0)) / np.ptp(X, axis=0) - 1

    fig, axs = plt.subplots(4, 4)

    for tau, ax in zip(np.linspace(0.01, 8, axs.size), axs.ravel()):
        theta, path = fit(X, y, tau, verbose=False)

        ax.scatter(X, y)

        w = np.linspace(X.min(), X.max(), 100)
        w_pred = predict(w, X, theta, tau)
        ax.plot(w, w_pred, color="red")
        ax.set_title(f"{tau:.3f} - {np.mean(np.square(w - w_pred)):.3f}")

    plt.show()


if __name__ == "__main__":
    _test()

import typing as t

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_ll(y, y_pred):
    return np.mean(
        y * np.log(1e-8 + y_pred) + (1 - y) * (1 - np.log(1e-8 + 1 - y_pred))
    )


def rel_err(err, prev_err):
    return (prev_err - err) / abs(1e-6 + prev_err)


def _add_bias(X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return np.hstack((np.ones((X.shape[0], 1)), X))


def predict(X, theta, add_bias: bool = True):
    if add_bias:
        X = _add_bias(X)

    return sigmoid(np.dot(theta, X.T))


def fit(
    X,
    y,
    step_size: float = 1e-2,
    max_it: int = 1e8,
    max_rel_err: float = 1e-9,
    random_state: t.Optional[int] = None,
    initial_theta: t.Optional[np.ndarray] = None,
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

    prev_log_likelihood = 1 / (1 + max_rel_err)
    gd_path = []

    while it < max_it and not np.isnan(prev_log_likelihood):
        it += 1
        y_pred = predict(X, theta, add_bias=False)
        err = y_pred - y
        theta -= step_size * np.mean(err * X.T, axis=1)

        # mse = np.mean(np.square(err))
        log_likelihood = calc_ll(y, y_pred)

        if it % it_to_print == 0:
            print(f"{it:<{9}} - MSE: {log_likelihood:.6f}")
            gd_path.append(np.hstack((theta, log_likelihood)))
            it_to_print = min(it_to_print + 5, 100)

        if abs(rel_err(log_likelihood, prev_log_likelihood)) <= max_rel_err:
            it = max_it

        prev_log_likelihood = log_likelihood

    print(f"Converged - MSE: {log_likelihood:.6f}")
    gd_path.append(np.hstack((theta, log_likelihood)))

    return theta, np.asarray(gd_path)


def _test():
    m = 75

    X = np.arange(m) + np.random.randn(m)

    y = np.logical_or(X > 0.7 * m, np.random.random(m) < 0.05)

    X = 2 * (X - X.min(axis=0)) / np.ptp(X, axis=0) - 1

    theta, path = fit(X, y)

    print("Parameters:", theta)

    def lin_predict(X, theta, add_bias: bool = True):
        if add_bias:
            X = _add_bias(X)
    
        return np.dot(theta, X.T)

    print("mean predict:", np.mean(predict(X, theta)))
    print("mean y:", np.mean(y))

    if X.ndim == 1:
        plt.scatter(X, y, color=["black" if cls >= 0.5 else "green" for cls in predict(X, theta)])

        plt.plot(X, lin_predict(X, theta), color="blue")

        w = np.linspace(X.min(), X.max(), 100)
        w_pred = predict(w, theta)

        plt.plot(w, w_pred, color="red")
        plt.title(f"LL: {calc_ll(w, w_pred):.6f}")

        plt.ylim(-0.1, 1.1)

        plt.show()

        # fig = plt.figure()
        # ax = fig.gca(projection="3d")
        # S1 = np.linspace(-40, 61, 100)
        # S1, S2 = np.meshgrid(S1, S1)

        # ERR = np.zeros_like(S1)

        # for i in np.arange(ERR.shape[0]):
        #     for j in np.arange(ERR.shape[1]):
        #         T1, T2 = S1[i, j], S2[i, j]
        #         ERR[i, j] = calc_ll(predict(X, [T1, T2]), y)

        # ax.plot_surface(S1, S2, ERR, cmap=matplotlib.cm.coolwarm, alpha=0.7)
        # ax.plot(path[:, 0], path[:, 1], path[:, 2], color="black")
        # ax.scatter(theta[0], theta[1], path[-1, 2], color="red", s=16)

        # plt.title("Error function surface")

        # plt.show()


if __name__ == "__main__":
    _test()

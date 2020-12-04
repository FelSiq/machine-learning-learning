import typing as t

import numpy as np

import train
import utils
import losses


def flatten_dict(values, remove_stats: bool = False):
    arrays = []
    shapes = {}
    cum_size = 0

    for param in sorted(values, key=lambda item: item[item[0] == "d" :]):
        is_grad = int(param.startswith("d"))
        if param[is_grad] not in {"A", "Z", "@"} and (not remove_stats or not param.startswith("moving")):
            v = values[param]
            shapes[param] = (v.shape, cum_size, cum_size + v.size)
            cum_size += v.size
            arrays.append(v.flatten())

    return np.concatenate(arrays), shapes


def unflatten_dict(theta, shapes):
    unfolded_params = {}

    for param, (shape, start, end) in shapes.items():
        unfolded_params[param] = theta[start:end].reshape(shape)

    return unfolded_params


def debug_forward(
    X,
    y,
    parameters: t.Dict[str, np.ndarray],
    grads: t.Dict[str, np.ndarray],
    lambd: float = 0.0,
    epsilon: float = 1e-7,
):
    vec_grads, grads_shapes = flatten_dict(grads)
    vec_theta, theta_shapes = flatten_dict(parameters)
    vec_gradapprox = np.zeros_like(vec_theta, dtype=float)

    for i in np.arange(vec_theta.size):
        vec_theta[i] += epsilon

        params_p = unflatten_dict(vec_theta, theta_shapes)

        A_p, _ = train.forward(X.T, params_p)
        loss_p, _ = losses.forward(A_p, y.T, params_p, "bce", lambd=lambd)

        vec_theta[i] -= 2 * epsilon

        params_m = unflatten_dict(vec_theta, theta_shapes)

        A_m, _ = train.forward(X.T, params_m)
        loss_m, _ = losses.forward(A_m, y.T, params_m, "bce", lambd=lambd)

        vec_theta[i] += epsilon

        vec_gradapprox[i] = (loss_p - loss_m) / (2 * epsilon)

    aux = unflatten_dict(vec_gradapprox, theta_shapes)
    vec_gradapprox, _ = flatten_dict({k: v for k, v in aux.items() if k[0] in {"W", "B", "b", "G"}})
    
    num = np.linalg.norm(vec_grads - vec_gradapprox)
    den = np.linalg.norm(vec_grads) + np.linalg.norm(vec_gradapprox)

    norm_diff = num / den

    assert norm_diff < 1e-6, norm_diff

    print(norm_diff)


def _test():
    import sklearn.datasets

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    y = y.reshape(-1, 1)

    model = utils.initialize_parameters((X.shape[1], 4, 1), "he")

    aux = unflatten_dict(*flatten_dict(model))

    assert all(np.allclose(model[k], aux[k]) for k in model if k in aux)

    lambd = 0

    A, caches = train.forward(X.T, model)
    loss, cache_l = losses.forward(A, y.T, model, "bce", lambd=lambd)
    caches.append(cache_l)
    grads = train.backward(A, caches)

    debug_forward(X, y, model, grads)


if __name__ == "__main__":
    _test()

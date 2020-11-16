import typing as t

import numpy as np

import modules
import utils
import losses


def forward(
    X,
    parameters,
    activation_hidden: str = "ReLU",
    activation_out: str = "sigmoid",
    keep_prob: float = 1.0,
):
    caches = []
    A = X
    L = len(parameters) // 2

    # Note: hidden layers forward
    for l in np.arange(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        A, cur_cache = modules.forward_linear_activation(
            A_prev=A,
            W=W,
            b=b,
            activation=activation_hidden,
        )
        caches.append(cur_cache)

        A, cur_cache = modules.forward_dropout(A, keep_prob)
        caches.append(cur_cache)

    # Note: output layer forward
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]

    A, cur_cache = modules.forward_linear_activation(
        A_prev=A,
        W=W,
        b=b,
        activation=activation_out,
    )

    caches.append(cur_cache)

    return A, caches


def backward(
    AL,
    caches,
    lambd: float = 0.0,
):
    assert lambd >= 0.0

    # Loss gradient
    cur_cache = caches.pop()
    dA = losses.backward(AL, cur_cache)

    # Note: L = (len(caches) - #output_layer) // #(paits dropout and linact) + #output_layer
    L = (len(caches) - 1) // 2 + 1
    m = AL.shape[1]
    grads = dict()

    # Output layer gradient
    cur_cache = caches.pop()
    dA, dW, db = modules.backward_linear_activation(dA, cur_cache, lambd=lambd)
    grads["dA" + str(L - 1)] = dA
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # Hidden layers gradient
    for l in np.arange(L - 2, -1, -1):
        cur_cache = caches.pop()
        dA = modules.backward_dropout(dA, cur_cache)

        cur_cache = caches.pop()
        dA, dW, db = modules.backward_linear_activation(dA, cur_cache, lambd=lambd)
        grads["dA" + str(l)] = dA
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate: float = 1e-2):
    assert learning_rate > 0.0

    L = len(parameters) // 2

    for l in np.arange(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters


def fit(
    X,
    y,
    parameters: t.Dict[str, t.Any],
    loss_func: str,
    lambd: float = 0.0,
    iterations: int = 100,
    it_to_print: int = -1,
):
    assert lambd >= 0.0

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    for i in np.arange(iterations):
        A, caches = forward(X.T, parameters)
        AL, cache_l = losses.forward(
            A, y.T, parameters, loss_func=loss_func, lambd=lambd
        )
        caches.append(cache_l)
        grads = backward(A, caches, lambd=lambd)
        parameters = update_parameters(parameters, grads)

        if it_to_print > 0 and i % it_to_print == 0:
            print(f"{i:<{8}}: {AL:.4f}")

    return parameters


def predict(X, parameters):
    preds, _ = forward(X.T, parameters, keep_prob=1.0)
    return np.squeeze(preds >= 0.5).astype(float, copy=False)


def _test():
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.preprocessing

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = utils.initialize_parameters([X.shape[1], 5, 1], "he")

    fit(X_train, y_train, model, "bce", iterations=1000, it_to_print=50, lambd=0.1)
    y_preds = predict(X_test, model)

    acc = sklearn.metrics.accuracy_score(y_preds, y_test)

    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    _test()

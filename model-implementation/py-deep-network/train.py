import typing as t

import numpy as np

import modules
import utils
import losses
import optimizers
import lr_updates


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


def update_parameters(
    parameters: t.Dict[str, np.ndarray],
    updates: t.Dict[str, np.ndarray],
    learning_rate: float = 1e-2,
):
    assert learning_rate > 0.0

    L = len(parameters) // 2

    for l in np.arange(L):
        parameters["W" + str(l + 1)] -= learning_rate * updates["uW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * updates["ub" + str(l + 1)]

    return parameters


def build_minibatch(
    X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, shuffle: bool = True
) -> t.Tuple[np.ndarray, np.ndarray]:
    num_inst = y.size

    inds = np.arange(num_inst)

    for _ in np.arange(epochs):
        cur_ind = 0

        if shuffle:
            np.random.shuffle(inds)

        while cur_ind < num_inst:
            inds_batch = inds[cur_ind : cur_ind + batch_size]

            X_batch = X[inds_batch, :]
            y_batch = y[inds_batch, :]

            cur_ind += batch_size

            yield X_batch, y_batch


def fit(
    X: np.ndarray,
    y: np.ndarray,
    parameters: t.Dict[str, t.Any],
    loss_func: str,
    optimizer: str = "gd",
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-2,
    shuffle_batch: bool = True,
    lambd: float = 0.0,
    optim_args: t.Optional[t.Dict[str, t.Any]] = None,
    lr_args: t.Optional[t.Dict[str, t.Any]] = None,
    lr_update: str = "constant",
    epoch_to_print: int = -1,
):
    assert lambd >= 0.0
    assert batch_size > 0

    batch_size = min(batch_size, y.size)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if optim_args is None:
        optim_args = dict()

    if lr_args is None:
        lr_args = dict()

    f_init_optim, f_update_optim = optimizers.solve_optim(optimizer)
    f_init_lr, f_update_lr = lr_updates.solve_lr_update(lr_update)

    cache_optim = f_init_optim(parameters, **optim_args)
    cache_lr = f_init_lr(learning_rate, **lr_args)

    inst_iterator = build_minibatch(
        X, y, epochs=epochs, batch_size=batch_size, shuffle=shuffle_batch
    )

    epoch = 0
    inst_ind = 0

    for X_batch, y_batch in inst_iterator:
        A, caches = forward(X_batch.T, parameters)
        AL, cache_l = losses.forward(
            A, y_batch.T, parameters, loss_func=loss_func, lambd=lambd
        )
        caches.append(cache_l)
        grads = backward(A, caches, lambd=lambd)
        updates = f_update_optim(grads, cache_optim)
        parameters = update_parameters(parameters, updates, learning_rate=learning_rate)
        learning_rate = f_update_lr(cache_lr)
        inst_ind += y_batch.size

        if epoch_to_print > 0 and inst_ind >= y.size:
            epoch += 1
            print(f"{epoch} / {epochs}: {AL:.4f}")
            inst_ind = 0

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

    fit(
        X_train,
        y_train,
        model,
        "bce",
        batch_size=128,
        optimizer="gd",
        epochs=50,
        learning_rate=0.5,
        epoch_to_print=50,
        lambd=0.1,
        lr_update="disc_frac",
        lr_args={"epochs": 10, "decay_rate": 0.75},
    )
    y_preds = predict(X_test, model)

    acc = sklearn.metrics.accuracy_score(y_preds, y_test)

    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    _test()

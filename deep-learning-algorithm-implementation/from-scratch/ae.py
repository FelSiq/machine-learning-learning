import typing as t

import numpy as np

import base
import modules
import optim
import losses


class Autoencoder(base.BaseModel):
    def __init__(
        self,
        dims: t.Sequence[int],
        learning_rate: float,
        first_momentum: float = 0.9,
        second_momentum: float = 0.999,
        clip_grad_norm: float = 1.0,
    ):
        assert len(dims) >= 3
        assert len(dims) % 2 == 1
        assert dims[0] == dims[-1]
        assert float(clip_grad_norm) > 0.0

        super(Autoencoder, self).__init__()

        self.layers = []
        l_rel = modules.ReLU()
        self.optim = optim.Adam(
            learning_rate=learning_rate,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
        )
        self.clip_grad_norm = float(clip_grad_norm)

        for i in range(1, len(dims)):
            l_lin = modules.Linear(dims[i - 1], dims[i])
            self.layers.append(l_lin)
            self.layers.append(l_rel)
            self.optim.register_layer(2 * (i - 1), *l_lin.parameters)

    def forward(self, X):
        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dout):
        if self.frozen:
            return

        for i, layer in enumerate(reversed(self.layers)):
            layer_id = len(self.layers) - i - 1
            grads = layer.backward(dout)
            self._clip_grads(grads)

            if not layer.trainable:
                dout = grads
                continue

            (dout,) = grads[0]
            param_grads = grads[1]
            grads = self.optim.update(layer_id, *param_grads)
            layer.update(*grads)


def _test():
    import matplotlib.pyplot as plt
    import sklearn.datasets
    import tqdm.auto

    np.random.seed(32)

    eval_size = 10
    train_size = 20000
    batch_size = 32
    train_epochs = 60
    learning_rate = 1e-3

    """
    X, _ = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    """
    X, _ = sklearn.datasets.load_digits(return_X_y=True)
    out_shape = (8, 8)
    np.random.shuffle(X)

    layer_dims = [
        int(ratio * X.shape[1]) for ratio in (1.0, 0.7, 0.2, 0.1, 0.2, 0.7, 1.0)
    ]
    layer_dims = [int(ratio * X.shape[1]) for ratio in (1.0, 0.75, 0.5, 0.75, 1.0)]

    X_eval, X_train = X[:eval_size, :], X[eval_size : train_size + eval_size, :]

    X_train_max = np.max(X_train)
    X_eval /= X_train_max
    X_train /= X_train_max

    print("Train shape :", X_train.shape)
    print("Eval shape  :", X_eval.shape)

    n = X_train.shape[0]
    model = Autoencoder(layer_dims, learning_rate)
    criterion = losses.MSELoss()

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss = 0.0
        it = 0
        np.random.shuffle(X)

        for start in tqdm.auto.tqdm(np.arange(0, n, batch_size)):
            end = start + batch_size
            X_batch = X_train[start:end, :]
            X_preds = model(X_batch)
            loss, loss_grad = criterion(X_batch, X_preds)
            model.backward(loss_grad)

            total_loss += loss
            it += 1

        total_loss /= it
        print(f"Total loss: {total_loss:.3f}")

    model.eval()
    X_preds = model(X_eval).astype(float)

    fig, axes = plt.subplots(
        2, eval_size, figsize=(15, 10), tight_layout=True, sharex=True, sharey=True
    )

    for i in np.arange(eval_size):
        ax_orig, ax_pred = axes[:, i]
        ax_orig.imshow(X_eval[i, :].reshape(*out_shape))
        ax_pred.imshow(X_preds[i, :].reshape(*out_shape))

    plt.show()


if __name__ == "__main__":
    _test()

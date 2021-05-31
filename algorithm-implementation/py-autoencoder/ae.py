import typing as t

import numpy as np

import modules
import optim
import losses


class Autoencoder:
    def __init__(
        self,
        dims: t.Sequence[int],
        learning_rate: float,
        momentum: float = 0.8,
        clip_grad_norm: float = 1.0,
    ):
        assert len(dims) >= 3
        assert len(dims) % 2 == 1
        assert dims[0] == dims[-1]
        assert float(clip_grad_norm) > 0.0

        self.layers = []
        l_rel = modules.ReLU()
        self.optim = optim.Momentum(learning_rate, momentum)
        self.clip_grad_norm = float(clip_grad_norm)

        for i in range(1, len(dims)):
            l_lin = modules.Linear(dims[i - 1], dims[i])
            self.layers.append(l_lin)
            self.layers.append(l_rel)
            self.optim.register_layer(2 * (i - 1), l_lin.weights, l_lin.bias)

    def _clip_grads(self, *grads):
        for grad in grads:
            np.clip(grad, -self.clip_grad_norm, self.clip_grad_norm, out=grad)

    def forward(self, X):
        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dout):
        for i, layer in enumerate(reversed(self.layers)):
            layer_id = len(self.layers) - i - 1
            grads = layer.backward(dout)

            if not layer.trainable:
                dout = grads
                continue

            self._clip_grads(*grads)

            dout = grads[0]
            param_grads = grads[1:]
            grads = self.optim.update(layer_id, *param_grads)
            layer.update(*grads)

    def __call__(self, X):
        return self.forward(X)

    def train(self):
        for layer in self.layers:
            layer.frozen = False

    def eval(self):
        for layer in self.layers:
            layer.frozen = True


def _test():
    import matplotlib.pyplot as plt
    import sklearn.datasets

    np.random.seed(16)

    eval_size = 5
    batch_size = 128
    train_epochs = 10
    learning_rate = 1e-2

    X, _ = sklearn.datasets.fetch_lfw_people(min_faces_per_person=5, return_X_y=True)
    print(X.shape)
    np.random.shuffle(X)

    layer_dims = [int(ratio * X.shape[1]) for ratio in (1.0, 0.75, 0.5, 0.75, 1.0)]

    X_eval, X_train = X[:eval_size, :], X[eval_size:, :]

    n = X_train.shape[0]
    model = Autoencoder(layer_dims, learning_rate)
    criterion = losses.MSELoss()

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss = 0.0
        np.random.shuffle(X)

        for start in np.arange(0, n, batch_size):
            end = start + batch_size
            X_batch = X_train[start:end, :]
            X_preds = model(X_batch)
            loss, loss_grad = criterion(X_batch, X_preds)
            model.backward(loss_grad)

            total_loss += loss

        total_loss /= batch_size
        print(f"Total loss: {total_loss:.3f}")

    model.eval()
    X_preds = model(X_eval).astype(int)

    fig, axes = plt.subplots(
        eval_size, 2, figsize=(10, 10), tight_layout=True, sharex=True, sharey=True
    )

    for i in np.arange(eval_size):
        ax_orig, ax_pred = axes[i]
        ax_orig.imshow(X_eval[i, :].reshape(62, 47))
        ax_pred.imshow(X_preds[i, :].reshape(62, 47))

    plt.show()


if __name__ == "__main__":
    _test()

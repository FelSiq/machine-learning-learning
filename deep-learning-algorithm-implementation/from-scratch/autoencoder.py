import typing as t

import numpy as np

import modules
import optimizers
import losses


class Autoencoder(modules.BaseModel):
    def __init__(self, dims: t.Sequence[int]):
        assert len(dims) >= 3
        assert len(dims) % 2 == 1
        assert dims[0] == dims[-1]

        super(Autoencoder, self).__init__()

        self.weights = modules.Sequential(
            [
                [
                    [
                        modules.Linear(
                            dims[i - 1],
                            dims[i],
                            activation=modules.ReLU(inplace=False),
                            include_bias=False,
                        ),
                        modules.BatchNorm1d(dims[i]),
                    ]
                    for i in range(1, len(dims) - 1)
                ],
                modules.Linear(
                    dims[-2], dims[-1], activation=modules.ReLU(inplace=True)
                ),
            ]
        )

        self.register_layers(self.weights)

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)


def _test():
    import matplotlib.pyplot as plt
    import sklearn.datasets
    import tqdm.auto

    np.random.seed(32)

    eval_size = 10
    train_size = 20000
    batch_size = 32
    train_epochs = 60
    learning_rate = 2e-3

    X, _ = sklearn.datasets.load_digits(return_X_y=True)
    out_shape = (8, 8)
    np.random.shuffle(X)

    layer_dims = [int(ratio * X.shape[1]) for ratio in (1.0, 0.75, 0.5, 0.75, 1.0)]

    X_eval, X_train = X[:eval_size, :], X[eval_size : train_size + eval_size, :]

    X_train_max = np.max(X_train)
    X_eval /= X_train_max
    X_train /= X_train_max

    print("Train shape :", X_train.shape)
    print("Eval shape  :", X_eval.shape)

    n = X_train.shape[0]
    model = Autoencoder(layer_dims)
    criterion = losses.MSELoss()

    optim = optimizers.Nadam(
        model.parameters,
        learning_rate=learning_rate,
        demon_iter_num=train_epochs / 1 * (n / batch_size),
        demon_min_mom=0.2,
    )

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss = 0.0
        it = 0
        np.random.shuffle(X)

        for start in tqdm.auto.tqdm(np.arange(0, n, batch_size)):
            optim.zero_grad()

            end = start + batch_size
            X_batch = X_train[start:end, :]

            X_preds = model(X_batch)
            loss, loss_grad = criterion(X_batch, X_preds)
            model.backward(loss_grad)

            optim.clip_grads_val()
            optim.step()

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

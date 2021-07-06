import typing as t

import numpy as np

import modules
import optimizers
import losses


class AttentionCNN(modules.BaseModel):
    def __init__(
        self,
        channels_num: t.Sequence[int],
        kernel_sizes: t.Sequence[int],
        linear_dims: t.Sequence[int],
        padding: str = "valid",
    ):
        assert len(channels_num) - 1 == len(kernel_sizes)
        super(AttentionCNN, self).__init__()

        self.weights = modules.Sequential(
            [
                [
                    [
                        modules.Conv2d(
                            channels_in=channels_num[i - 1],
                            channels_out=channels_num[i],
                            kernel_size=kernel_sizes[i - 1],
                            activation=None,
                            padding_type=padding,
                            include_bias=False,
                        ),
                        modules.BatchNorm2d(channels_num[i]),
                        modules.ReLU(inplace=True),
                        # modules.SqueezeExcite(channels_num[i]),
                        modules.ConvBlockAttention2d(
                            channels_in=channels_num[i],
                            kernel_size=kernel_sizes[i - 1],
                            spatial_norm_layer_after_conv=modules.BatchNorm2d(1),
                        ),
                        modules.SpatialDropout(drop_prob=0.1, inplace=True),
                    ]
                    for i in range(1, len(channels_num))
                ],
                modules.Flatten(),
                [
                    [
                        modules.Linear(
                            linear_dims[i - 1],
                            linear_dims[i],
                            activation=None,
                            include_bias=False,
                        ),
                        modules.BatchNorm1d(linear_dims[i]),
                        modules.ReLU(inplace=True),
                        modules.Dropout(drop_prob=0.3, inplace=True),
                    ]
                    for i in range(1, len(linear_dims) - 1)
                ],
                modules.Linear(
                    linear_dims[-2],
                    linear_dims[-1],
                    activation=None,
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

    batch_size = 512
    train_epochs = 2
    learning_rate = 1e-4

    channels_num = (1, 8, 8)
    kernel_sizes = (3, 3)

    linear_dims = (4608, 128, 10)

    X, y = sklearn.datasets.fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    X += 64 * np.random.randn(*X.shape)
    X = (X / 255.0).reshape(-1, 28, 28, 1)

    inds = np.arange(y.size)
    X = X[inds, :]
    y = y[inds]

    X_train, X_eval, X_test = X[:5000], X[5000:6000], X[6000:7000]
    y_train, y_eval, y_test = y[:5000], y[5000:6000], y[6000:7000]

    print("Train shape :", X_train.shape)
    print("Eval shape  :", X_eval.shape)
    print("Test shape  :", X_test.shape)

    n = X_train.shape[0]

    model = AttentionCNN(channels_num, kernel_sizes, linear_dims)
    criterion = losses.CrossEntropyLoss()
    optim = optimizers.Nadam(model.parameters, learning_rate=learning_rate)

    inds = np.arange(X_train.shape[0])

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_train = total_loss_eval = 0.0
        it = 0

        np.random.shuffle(inds)
        X_train = X_train[inds, :]
        y_train = y_train[inds]

        for start in tqdm.auto.tqdm(np.arange(0, n, batch_size)):
            optim.zero_grad()

            end = start + batch_size
            X_batch = X_train[start:end, :]
            y_batch = y_train[start:end]

            model.train()
            y_logits = model(X_batch)
            loss, loss_grad = criterion(y_batch, y_logits)
            model.backward(loss_grad)
            total_loss_train += loss

            optim.clip_grads_norm()
            optim.step()

            model.eval()
            y_logits = model(X_eval)
            loss, _ = criterion(y_eval, y_logits)
            total_loss_eval += loss

            it += 1

        total_loss_train /= it
        total_loss_eval /= it

        print(f"Total loss (train) : {total_loss_train:.3f}")
        print(f"Total loss (eval)  : {total_loss_eval:.3f}")

    model.train()
    y_preds_logits = model(X_test)
    loss, loss_grad = criterion(y_test, y_preds_logits)
    dX = model.backward(loss_grad)
    y_preds = y_preds_logits.argmax(axis=-1)
    test_acc = float(np.mean(y_preds == y_test))
    print(f"Test acc: {test_acc:.3f}")

    X_test_plot = X_test[:10, ...]
    dX_test_plot = dX[:10, ...]
    fig, axes = plt.subplots(
        3, X_test_plot.shape[0], figsize=(15, 10), sharex=True, sharey=True
    )

    for i, (X, dX) in enumerate(zip(X_test_plot, dX_test_plot)):
        A = np.squeeze(X)
        B = np.squeeze(dX)
        W = A * (B - np.min(B)) / (1e-7 + np.ptp(B))
        axes[0][i].imshow(A, cmap="hot")
        axes[1][i].imshow(B, cmap="gray")
        axes[2][i].imshow(W, cmap="hot")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _test()

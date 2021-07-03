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
        padding: str = "same",
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

    eval_size = 128
    test_size = 128
    batch_size = 64
    train_epochs = 5
    learning_rate = 1e-4
    padding = "valid"

    channels_num = (1, 32, 32)
    kernel_sizes = (2, 2) if padding == "valid" else (3, 3)

    linear_dims = (
        (6 * 6 * channels_num[-1], 512, 10)
        if padding == "valid"
        else (8 * 8 * channels_num[-1], 512, 10)
    )

    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8, 1)

    out_shape = (8, 8)

    inds = np.arange(X.shape[0])
    np.random.shuffle(inds)
    X = X[inds, :]
    y = y[inds]

    X_test, X = X[:test_size, :], X[test_size:, :]
    y_test, y = y[:test_size], y[test_size:]

    X_eval, X_train = X[:eval_size, :], X[eval_size:, :]
    y_eval, y_train = y[:eval_size], y[eval_size:]

    X_train_max = np.max(X_train)

    X_train /= X_train_max
    X_eval /= X_train_max
    X_test /= X_train_max

    print("Train shape :", X_train.shape)
    print("Eval shape  :", X_eval.shape)

    n = X_train.shape[0]
    model = AttentionCNN(channels_num, kernel_sizes, linear_dims, padding=padding)
    criterion = losses.CrossEntropyLoss()

    optim = optimizers.RMSProp(
        model.parameters, learning_rate=learning_rate, clip_grad_val=0.1
    )

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

            optim.clip_grads_val()
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

    model.eval()
    y_preds_logits = model(X_test)
    y_preds = y_preds_logits.argmax(axis=-1)
    test_acc = float(np.mean(y_preds == y_test))
    print(f"Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    _test()

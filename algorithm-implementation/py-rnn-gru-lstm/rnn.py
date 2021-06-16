import numpy as np

import losses
import modules
import optim


class RNN:
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        learning_rate: float,
        clip_grad_norm: float = 1.0,
    ):
        assert float(clip_grad_norm) > 0.0
        self.rnn_cell = modules.RNNCell(dim_in, dim_hidden, dim_out)
        self.optim = optim.Nadam(learning_rate)
        self.clip_grad_norm = float(clip_grad_norm)

        self.dim_hidden = dim_hidden

        self.optim.register_layer(
            0,
            self.rnn_cell.lin_hidden.weights,
            self.rnn_cell.lin_input.weights,
            self.rnn_cell.lin_input.bias,
            self.rnn_cell.lin_outut.weights,
            self.rnn_cell.lin_outut.bias,
        )

        self.total_timesteps = 0

    def _clip_grads(self, *grads):
        for grad in grads:
            np.clip(grad, -self.clip_grad_norm, self.clip_grad_norm, out=grad)

    def forward(self, X):
        # X.shape = (time, batch, dim)
        outputs = np.empty((X.shape[0], X.shape[1], dim_out), dtype=float)

        for t, timestep in enumerate(X):
            outputs[t, :] = self.rnn_cell.forward(out)
            self.total_timesteps += 1

        return outputs

    def backward(self, dout_y):
        # Fix this.
        dout_y = dout
        dout_h = np.zeros(self.dim_hidden, dtype=float)

        while self.total_timesteps > 0:
            grads = layer.backward(dout_y, dout_h)
            self._clip_grads(*grads)

            if not layer.trainable:
                dout = grads[0]
                continue

            dout = grads[0]
            param_grads = grads[1:]
            grads = self.optim.update(0, *param_grads)
            layer.update(*grads)

            self.total_timesteps -= 1

    def __call__(self, X):
        return self.forward(X)

    def train(self):
        for layer in self.layers:
            layer.frozen = False

    def eval(self):
        for layer in self.layers:
            layer.frozen = True


def _test():
    import pandas as pd
    import tqdm.auto

    np.random.seed(32)

    eval_size = 10
    train_size = 20000
    batch_size = 32
    train_epochs = 60
    learning_rate = 1e-3

    X = np.linspace(-2, 2, 100).reshape(100, 1, 1)

    np.random.shuffle(X)

    X_eval, X_train = X[:eval_size, :], X[eval_size : train_size + eval_size, :]

    print("Train shape :", X_train.shape)
    print("Eval shape  :", X_eval.shape)

    n = X_train.shape[0]
    model = RNN(, learning_rate)
    criterion = losses.MSELoss()

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss = 0.0
        np.random.shuffle(X)

        for start in tqdm.auto.tqdm(np.arange(0, n, batch_size)):
            end = start + batch_size
            X_batch = X_train[start:end, :]
            X_preds = model(X_batch)
            loss, loss_grad = criterion(X_batch, X_preds)
            model.backward(loss_grad)

            total_loss += loss

        total_loss /= batch_size
        print(f"Total loss: {total_loss:.3f}")

    model.eval()
    X_preds = model(X_eval)


if __name__ == "__main__":
    _test()

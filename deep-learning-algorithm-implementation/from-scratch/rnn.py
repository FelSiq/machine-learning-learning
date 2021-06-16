import numpy as np

import base
import losses
import modules
import optim


class RNN(base.BaseModel):
    def __init__(
        self,
        num_embed_tokens: int,
        dim_embed: int,
        dim_hidden: int,
        dim_out: int,
        learning_rate: float,
        clip_grad_norm: float = 1.0,
    ):
        assert float(clip_grad_norm) > 0.0

        self.embed_layer = modules.Embedding(num_embed_tokens, dim_embed)
        self.rnn_cell = modules.RNNCell(dim_embed, dim_hidden, dim_out)
        self.optim = optim.Nadam(learning_rate)
        self.clip_grad_norm = float(clip_grad_norm)

        self.dim_hidden = int(dim_hidden)

        self.optim.register_layer(0, *self.embed_layer.parameters)
        self.optim.register_layer(1, *self.rnn_cell.parameters)

    def forward(self, X):
        # X.shape = (time, batch)
        outputs = np.empty((X.shape[0], X.shape[1], dim_out), dtype=float)

        X_embedded = self.embed_layer(X)

        for t, X_t in enumerate(X_embedded):
            outputs[t, :] = self.rnn_cell.forward(X_t)

        return outputs

    def backward(self, dout_y):
        # if n_dim = 3: dout_y (time, batch, dim_out)
        # if n_dim = 2: dout_y (batch, dim_out) (only the last timestep)

        batch_size = dout_y.shape[1] if dout_y.ndim == 3 else dout_y.shape[0]
        cur_dout_y = dout_y
        dout_h = np.zeros((batch_size, self.dim_hidden), dtype=float)
        douts_X = []  # type: t.List[np.ndarray]
        i = 0

        while self.rnn_cel.has_stored_grads:
            grads = layer.backward(dout_h, cur_dout_y)
            self._clip_grads(*grads)

            dout_h, dout_X = grads[:2]
            param_grads = grads[2:]

            douts_X.insert(0, dout_X)

            if dout_y.ndim == 3:
                cur_dout_y = dout_y[-i - 1]

            elif i == 0:
                cur_dout_y = np.zeros((batch_size, self.dim_hidden), dtype=float)

            grads = self.optim.update(1, *param_grads)
            layer.update(*grads)
            i += 1

        douts_X = np.asfarray(douts_X)
        # shape: (time, batch, emb_dim)

        return douts_X


def _test():
    import pandas as pd
    import tqdm.auto
    import tweets_utils

    np.random.seed(32)

    eval_size = 10
    train_size = 20000
    batch_size = 32
    train_epochs = 60
    learning_rate = 1e-3

    X_train, y_train, X_test, y_test, _, _ = tweets_utils.get_data(1000)
    print(X_train[:3])
    exit(0)

    np.random.shuffle(X)

    X_eval, X_train = X[:eval_size, :], X[eval_size : train_size + eval_size, :]

    print("Train shape :", X_train.shape)
    print("Eval shape  :", X_eval.shape)

    n = X_train.shape[0]
    model = None  # RNN(, learning_rate)
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

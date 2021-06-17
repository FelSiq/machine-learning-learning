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
        self.rnn_cell = modules.RNNCell(dim_embed, dim_hidden)
        self.lin_out_layer = modules.Linear(dim_hidden, dim_out)

        self.optim = optim.Nadam(learning_rate)
        self.clip_grad_norm = float(clip_grad_norm)

        self.dim_embed = int(dim_embed)
        self.dim_hidden = int(dim_hidden)
        self.dim_out = int(dim_out)

        self.layers = (
            self.embed_layer,
            self.rnn_cell,
            self.lin_out_layer,
        )

        self.optim.register_layer(0, *self.embed_layer.parameters)
        self.optim.register_layer(1, *self.rnn_cell.parameters)
        self.optim.register_layer(2, *self.lin_out_layer.parameters)

    def forward(self, X):
        # X.shape = (time, batch)
        outputs = np.empty((X.shape[0], X.shape[1], self.dim_out), dtype=float)

        X_embedded = self.embed_layer(X)

        for t, X_t in enumerate(X_embedded):
            cur_out = self.rnn_cell.forward(X_t)
            cur_out = self.lin_out_layer.forward(cur_out)
            outputs[t, ...] = cur_out

        return outputs

    def backward(self, douts):
        # if n_dim = 3: dout_y (time, batch, dim_out)
        # if n_dim = 2: dout_y (batch, dim_out) (only the last timestep)

        if douts.ndim == 2:
            douts = np.expand_dims(douts, 0)

        batch_size = douts.shape[1]

        douts_X = []  # type: t.List[np.ndarray]
        douts_y = []  # type: t.List[np.ndarray]

        for dout in reversed(douts):
            grads = self.lin_out_layer.backward(dout)
            self._clip_grads(*grads)

            dout_y = grads[0]
            param_grads = grads[1:]
            douts_y.insert(0, dout_y)

            grads = self.optim.update(2, *param_grads)
            self.lin_out_layer.update(*grads)

        self.lin_out_layer.clean_grad_cache()
        douts_y = np.asfarray(douts_y)

        # if n_dim = 3: douts_y (time, batch, dim_hidden)
        # if n_dim = 2: douts_y (batch, dim_hidden) (only the last timestep)
        dout_h = np.zeros((batch_size, self.dim_hidden), dtype=float)
        dout_y = douts_y[-1] if douts_y.ndim == 3 else douts_y
        i = 1

        while self.rnn_cell.has_stored_grads:
            grads = self.rnn_cell.backward(dout_h, dout_y)
            self._clip_grads(*grads)

            dout_h, dout_X = grads[:2]
            param_grads = grads[2:]

            douts_X.insert(0, dout_X)

            if dout_y.ndim == 3:
                dout_y = douts_y[-i]

            elif i == 1:
                dout_y = np.zeros((batch_size, self.dim_hidden), dtype=float)

            grads = self.optim.update(1, *param_grads)
            self.rnn_cell.update(*grads)
            i += 1

        self.rnn_cell.reset()
        self.rnn_cell.clean_grad_cache()
        douts_X = np.asfarray(douts_X)
        # shape: (time, batch, emb_dim)

        for dout_X in reversed(douts_X):
            grads = self.embed_layer.backward(dout_X)
            self._clip_grads(*grads)

            dout = grads[0]
            param_grads = grads[1:]

            grads = self.optim.update(0, *param_grads)
            self.embed_layer.update(*grads)

        self.embed_layer.clean_grad_cache()


def _test():
    import pandas as pd
    import tqdm.auto
    import tweets_utils

    np.random.seed(32)

    batch_size = 32
    train_epochs = 5

    X_train, y_train, X_test, y_test, word_count = tweets_utils.get_data()

    token_dictionary = tweets_utils.build_dictionary(word_count, max_token_num=512)
    tweets_utils.encode_tweets(X_train, token_dictionary)
    tweets_utils.encode_tweets(X_test, token_dictionary)

    model = RNN(
        num_embed_tokens=len(token_dictionary) + 1,
        dim_embed=16,
        dim_hidden=64,
        dim_out=1,
        learning_rate=1e-2,
    )

    criterion = losses.BCELoss(with_logits=True)

    def pad_batch(X):
        batch_max_seq_len = int(max(map(len, X)))

        X_padded = np.array(
            [
                np.hstack(
                    (
                        np.array(inst, dtype=int),
                        np.zeros(batch_max_seq_len - len(inst), dtype=int),
                    )
                )
                for inst in X
            ],
            dtype=int,
        )

        return X_padded

    batch_inds = np.arange(len(X_train))

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss = 0.0
        np.random.shuffle(batch_inds)
        X_train = [X_train[i] for i in batch_inds]
        y_train = y_train[batch_inds]

        for start in tqdm.auto.tqdm(np.arange(0, len(X_train), batch_size)):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            X_batch = pad_batch(X_batch)

            y_logits = model(X_batch.T)
            y_logits = y_logits[-1]

            loss, loss_grad = criterion(y_batch, y_logits)
            model.backward(loss_grad)

            total_loss += loss

        total_loss /= batch_size
        print(f"Total loss: {total_loss:.3f}")

    model.eval()
    X_test = pad_batch(X_test)
    y_preds_logits = model(X_test.T)
    test_acc = float(np.mean((y_preds_logits > 0.0) == y_test))
    print(f"Eval acc: {test_acc:.3f}")


if __name__ == "__main__":
    _test()

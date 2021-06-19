import numpy as np

import base
import losses
import modules
import optim


class SequenceModel(base.BaseModel):
    def __init__(
        self,
        cell_model,
        learning_rate: float,
        clip_grad_norm: float = 1.0,
    ):
        assert float(clip_grad_norm) > 0.0

        super(SequenceModel, self).__init__()

        self.sequence_cell = cell_model

        self.optim = optim.Nadam(learning_rate)
        self.clip_grad_norm = float(clip_grad_norm)

        self.dim_in = int(self.sequence_cell.dim_in)
        self.dim_hidden = int(self.sequence_cell.dim_hidden)

        self.uses_cell_state = isinstance(cell_model, modules.LSTMCell)

        self.layers = (self.sequence_cell,)
        self.parameters = (*self.sequence_cell.parameters,)

        self.optim.register_layer(0, *self.sequence_cell.parameters)

    def forward(self, X):
        # X.shape = (time, batch)
        outputs = np.empty((X.shape[0], X.shape[1], self.dim_hidden), dtype=float)

        for t, X_t in enumerate(X):
            outputs[t, ...] = self.sequence_cell(X_t)

        self.sequence_cell.reset()

        return outputs

    def backward(self, douts):
        if douts.ndim == 2:
            douts = np.expand_dims(douts, 0)

        # TODO: fix this method for LSTMs (which return both cell state and hidden state)

        batch_size = douts.shape[1]

        douts_X = []  # type: t.List[np.ndarray]
        douts = np.squeeze(douts)

        # if n_dim = 3: douts (time, batch, dim_hidden)
        # if n_dim = 2: douts (batch, dim_hidden) (only the last timestep)

        if self.uses_cell_state:
            dout_cs = np.zeros((batch_size, self.dim_hidden), dtype=float)

        dout_h = np.zeros((batch_size, self.dim_hidden), dtype=float)
        dout = douts[-1] if douts.ndim == 3 else douts
        i = 1

        while self.sequence_cell.has_stored_grads:
            if self.uses_cell_state:
                grads = self.sequence_cell.backward(dout_h + dout, dout_cs)
                self._clip_grads(grads)
                (dout_h, dout_cs, dout_X) = grads[0]

            else:
                grads = self.sequence_cell.backward(dout_h + dout)
                self._clip_grads(grads)
                (dout_h, dout_X) = grads[0]

            param_grads = grads[1]

            douts_X.insert(0, dout_X)

            if dout.ndim == 3:
                dout = douts[-i - 1]

            elif i == 1:
                dout = np.zeros_like(dout)

            grads = self.optim.update(0, *param_grads)
            self.sequence_cell.update(*grads)
            i += 1

        self.sequence_cell.clean_grad_cache()
        douts_X = np.asfarray(douts_X)
        # shape: (time, batch, dim_in)

        return douts_X


class RNN(SequenceModel):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        learning_rate: float,
        clip_grad_norm: float = 1.0,
    ):
        rnn_cell = modules.RNNCell(dim_in, dim_hidden)
        super(RNN, self).__init__(rnn_cell, learning_rate, clip_grad_norm)


class GRU(SequenceModel):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        learning_rate: float,
        clip_grad_norm: float = 1.0,
    ):
        gru_cell = modules.GRUCell(dim_in, dim_hidden)
        super(GRU, self).__init__(gru_cell, learning_rate, clip_grad_norm)


class LSTM(SequenceModel):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        learning_rate: float,
        clip_grad_norm: float = 1.0,
    ):
        lstm_cell = modules.LSTMCell(dim_in, dim_hidden)
        super(LSTM, self).__init__(lstm_cell, learning_rate, clip_grad_norm)


class DeepSequenceModel(base.BaseModel):
    def __init__(self, model, num_layers: int):
        assert int(num_layers) > 0

        super(DeepSequenceModel, self).__init__()

        self.weights = modules.Sequential([model.copy() for _ in range(num_layers)])

        for model in self.weights[1:]:
            # TODO: change input dim of middle dimensions here
            pass

        self.parameters = (*self.weights.parameters,)
        self.layers = (self.weights,)

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)


class Bidirectional(base.BaseModel):
    def __init__(self, model):
        super(Bidirectional, self).__init__()

        self.rnn_l_to_r = model
        self.rnn_r_to_l = model.copy()

        self.dim_hidden = int(self.rnn_l_to_r.dim_hidden)

        self.layers = (
            self.rnn_l_to_r,
            self.rnn_r_to_l,
        )

        self.parameters(
            *self.rnn_l_to_r.parameters,
            *self.rnn_r_to_l.parameters,
        )

    def forward(self, X):
        # shape: (time, batch, dim_in)
        out_l_to_r = self.rnn_l_to_r(X)
        out_r_to_l = self.rnn_r_to_l(X[::-1, ...])
        # shape (both): (time, batch, dim_hidden)

        outputs = np.dstack((out_l_to_r, out_r_to_l))
        # shape: (time, batch, 2 * dim_hidden)

        return outputs

    def backward(self, douts):
        # if n_dim = 3: douts (time, batch, 2 * dim_hidden)
        # if n_dim = 2: douts (batch, 2 * dim_hidden)
        douts_l_to_r = self.rnn_l_to_r.backward(douts[..., : self.dim_hidden])
        douts_r_to_l = self.rnn_r_to_l.backward(douts[..., self.dim_hidden :])

        douts_X = douts_l_to_r + douts_r_to_l

        return douts_X


class NLPProcessor(base.BaseModel):
    def __init__(
        self,
        num_embed_tokens: int,
        dim_embed: int,
        dim_hidden: int,
        dim_out: int,
        learning_rate: float,
        clip_grad_norm: float = 1.0,
        bidirectional: bool = False,
        num_layers: int = 1,
    ):
        assert float(clip_grad_norm) > 0.0

        super(NLPProcessor, self).__init__()

        self.embed_layer = modules.Embedding(num_embed_tokens, dim_embed)
        self.rnn = LSTM(
            dim_embed, dim_hidden, learning_rate, clip_grad_norm=clip_grad_norm
        )

        if bidirectional:
            self.rnn = Bidirectional(self.rnn)

        if int(num_layers) > 1:
            self.rnn = DeepSequenceModel(self.rnn, num_layers)

        self.lin_out_layer = modules.Linear(
            dim_hidden * (1 + int(bool(bidirectional))), dim_out
        )

        self.optim = optim.Nadam(learning_rate)
        self.clip_grad_norm = float(clip_grad_norm)

        self.dim_embed = int(dim_embed)
        self.dim_hidden = int(dim_hidden)
        self.dim_out = int(dim_out)

        self.layers = (
            self.embed_layer,
            self.rnn,
            self.lin_out_layer,
        )

        self.parameters = (
            *self.embed_layer.parameters,
            *self.rnn.parameters,
            *self.lin_out_layer,
        )

        self.optim.register_layer(0, *self.embed_layer.parameters)
        self.optim.register_layer(1, *self.lin_out_layer.parameters)

    def forward(self, X):
        # X.shape = (time, batch)
        X_embedded = self.embed_layer(X)
        outputs = self.rnn(X_embedded)

        outputs_b = np.empty((X.shape[0], X.shape[1], self.dim_out), dtype=float)

        for t, out_t in enumerate(outputs):
            outputs_b[t, ...] = self.lin_out_layer(out_t)

        return outputs_b

    def backward(self, douts):
        # if n_dim = 3: dout_y (time, batch, dim_out)
        # if n_dim = 2: dout_y (batch, dim_out) (only the last timestep)

        if douts.ndim == 2:
            douts = np.expand_dims(douts, 0)

        batch_size = douts.shape[1]

        douts_y = []  # type: t.List[np.ndarray]

        for dout in reversed(douts):
            grads = self.lin_out_layer.backward(dout)
            self._clip_grads(grads)

            (dout_y,) = grads[0]
            param_grads = grads[1]
            douts_y.insert(0, dout_y)

            grads = self.optim.update(1, *param_grads)
            self.lin_out_layer.update(*grads)

        self.lin_out_layer.clean_grad_cache()
        douts_y = np.squeeze(np.asfarray(douts_y))

        douts_X = self.rnn.backward(douts_y)
        # shape: (time, batch, emb_dim)

        grads = self.embed_layer.backward(douts_X)
        self._clip_grads(grads)

        param_grads = grads[1]

        grads = self.optim.update(0, *param_grads)
        self.embed_layer.update(*grads)

        self.embed_layer.clean_grad_cache()


def _test():
    import pandas as pd
    import tqdm.auto
    import tweets_utils

    def pad_batch(X):
        lens = np.fromiter(map(len, X), count=len(X), dtype=int)
        batch_max_seq_len = int(max(lens))

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

    np.random.seed(32)

    batch_size = 32
    train_epochs = 10

    X_train, y_train, X_test, y_test, word_count = tweets_utils.get_data()

    y_train = y_train.ravel().astype(int, copy=False)
    y_test = y_test.ravel().astype(int, copy=False)

    X_eval, X_test = X_test[:50], X_test[50:]
    y_eval, y_test = y_test[:50], y_test[50:]

    token_dictionary = tweets_utils.build_dictionary(word_count, max_token_num=2048)
    tweets_utils.encode_tweets(X_train, token_dictionary)
    tweets_utils.encode_tweets(X_test, token_dictionary)
    tweets_utils.encode_tweets(X_eval, token_dictionary)

    X_test = pad_batch(X_test)
    X_eval = pad_batch(X_eval)

    model = NLPProcessor(
        num_embed_tokens=1 + len(token_dictionary),
        dim_embed=16,
        dim_hidden=64,
        dim_out=1,
        learning_rate=2.5e-4,
        bidirectional=False,
        num_layers=2,
    )

    criterion = losses.BCELoss(with_logits=True)

    batch_inds = np.arange(len(X_train))

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_train = total_loss_eval = 0.0
        it = 0

        np.random.shuffle(batch_inds)
        X_train = [X_train[i] for i in batch_inds]
        y_train = y_train[batch_inds]

        for start in tqdm.auto.tqdm(np.arange(0, len(X_train), batch_size)):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            X_batch = pad_batch(X_batch)

            model.train()
            y_logits = model(X_batch.T)
            y_logits = y_logits[-1]
            loss, loss_grad = criterion(y_batch, y_logits)
            model.backward(loss_grad)
            total_loss_train += loss

            model.eval()
            y_logits = model(X_eval.T)
            y_logits = y_logits[-1]
            loss, loss_grad = criterion(y_eval, y_logits)
            total_loss_eval += loss

            it += 1

        total_loss_train /= it
        total_loss_eval /= it

        print(f"Total loss (train) : {total_loss_train:.3f}")
        print(f"Total loss (eval)  : {total_loss_eval:.3f}")

    model.eval()
    y_preds_logits = model(X_test.T)

    if isinstance(criterion, losses.BCELoss):
        y_preds_logits = np.squeeze(y_preds_logits[-1])
        y_preds = (y_preds_logits > 0.0).astype(int, copy=False)

    else:
        y_preds_logits = y_preds_logits[-1]
        y_preds = y_preds_logits.argmax(axis=-1)

    test_acc = float(np.mean(y_preds == y_test))
    print(f"Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    _test()

import numpy as np

import losses
import modules
import optimizers


class _BaseSequenceModel(modules.BaseModel):
    def __init__(self, cell_model):
        super(_BaseSequenceModel, self).__init__()

        self.sequence_cell = cell_model

        self.dim_in = int(self.sequence_cell.dim_in)
        self.dim_hidden = int(self.sequence_cell.dim_hidden)

        self.register_layers(self.sequence_cell)

        self.uses_cell_state = False

    def forward(self, X):
        dim_time, dim_batch, _ = X.shape
        outputs = np.empty((dim_time, dim_batch, self.dim_hidden), dtype=float)

        for t, X_t in enumerate(X):
            outputs[t, ...] = self.sequence_cell(X_t)

        self.sequence_cell.reset()

        return outputs

    def backward(self, douts):
        dim_time, dim_batch, dim_hidden = douts.shape

        douts_X = []  # type: t.List[np.ndarray]

        if self.uses_cell_state:
            dout_cs = np.zeros((dim_batch, self.dim_hidden), dtype=float)

        dout = douts[-1]
        i = 1

        while self.sequence_cell.has_stored_grads:
            if self.uses_cell_state:
                (dout_h, dout_cs, dout_X) = self.sequence_cell.backward(dout, dout_cs)

            else:
                (dout_h, dout_X) = self.sequence_cell.backward(dout)

            douts_X.insert(0, dout_X)
            dout = dout_h

            if i < dim_time:
                dout += douts[-i - 1]
                i += 1

        self.sequence_cell.clean_grad_cache()
        douts_X = np.asfarray(douts_X)
        # shape: (time, batch, dim_in)

        return douts_X


class RNN(_BaseSequenceModel):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(RNN, self).__init__(modules.RNNCell(dim_in, dim_hidden))


class GRU(_BaseSequenceModel):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(GRU, self).__init__(modules.GRUCell(dim_in, dim_hidden))


class LSTM(_BaseSequenceModel):
    def __init__(self, dim_in: int, dim_hidden: int):
        super(LSTM, self).__init__(modules.LSTMCell(dim_in, dim_hidden))
        self.uses_cell_state = True


class SequenceProcessor(modules.BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        bidirectional: bool = False,
        num_layers: int = 1,
        cell_type: _BaseSequenceModel = GRU,
    ):
        assert cell_type in {RNN, GRU, LSTM}

        super(SequenceProcessor, self).__init__()

        self.rnn = cell_type(dim_in, dim_hidden)

        if int(num_layers) > 1:
            self.rnn = modules.DeepSequenceModel(self.rnn, num_layers)

        if bidirectional:
            self.rnn = modules.Bidirectional(self.rnn)

        self.lin_out_layer = modules.Linear(
            dim_hidden * (1 + int(bool(bidirectional))), dim_out
        )

        self.dim_in = int(dim_in)
        self.dim_hidden = int(dim_hidden)
        self.dim_out = int(dim_out)

        self.register_layers(self.rnn, self.lin_out_layer)

    def forward(self, X):
        if X.ndim != 3:
            X = np.expand_dims(X, -1)

        dim_time, dim_batch, _ = X.shape
        outputs = self.rnn(X)
        outputs_b = np.empty((dim_time, dim_batch, self.dim_out), dtype=float)

        for t, out_t in enumerate(outputs):
            outputs_b[t, ...] = self.lin_out_layer(out_t)

        return outputs_b

    def backward(self, douts):
        dim_time, dim_batch, dim_out = douts.shape

        douts_y = []  # type: t.List[np.ndarray]

        for dout in reversed(douts):
            dout_y = self.lin_out_layer.backward(dout)
            douts_y.insert(0, dout_y)

        douts_y = np.asfarray(douts_y)
        douts_X = self.rnn.backward(douts_y)
        # shape: (time, batch, emb_dim)

        self.lin_out_layer.clean_grad_cache()

        return douts_X


class NLPProcessor(modules.BaseModel):
    def __init__(
        self,
        num_embed_tokens: int,
        dim_embed: int,
        dim_hidden: int,
        dim_out: int,
        bidirectional: bool = False,
        num_layers: int = 1,
        cell_type: _BaseSequenceModel = GRU,
    ):
        super(NLPProcessor, self).__init__()

        self.embed_layer = modules.Embedding(num_embed_tokens, dim_embed)

        self.sequence_processor = SequenceProcessor(
            dim_in=dim_embed,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            bidirectional=bidirectional,
            num_layers=num_layers,
            cell_type=cell_type,
        )

        self.register_layers(self.embed_layer, self.sequence_processor)

    def forward(self, X):
        # X.shape = (time, batch)
        X_embedded = self.embed_layer(X)
        out = self.sequence_processor(X_embedded)
        return out

    def backward(self, douts):
        douts = self.sequence_processor.backward(douts)
        self.embed_layer.backward(douts)


def _test_nlp():
    import tqdm.auto
    from test import tweets_utils

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
    train_epochs = 5

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
        bidirectional=False,
        num_layers=2,
        cell_type=LSTM,
    )

    criterion = losses.BCELoss(with_logits=True)
    optim = optimizers.Nadam(model.parameters, learning_rate=2.5e-3)

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
            loss_grad = np.expand_dims(loss_grad, 0)
            model.backward(loss_grad)
            total_loss_train += loss

            optim.clip_grads_val()
            optim.step()

            model.eval()
            y_logits = model(X_eval.T)
            y_logits = y_logits[-1]
            loss, _ = criterion(y_eval, y_logits)
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


def _test_forecasting():
    import tqdm.auto
    from test import time_series_utils

    np.random.seed(32)

    batch_size = 128
    train_epochs = 3

    X_train, X_test = time_series_utils.get_data(nrows=3000)
    X_eval, X_test = X_test[:50], X_test[50:]

    y_test = X_test[:, 1:].T
    y_eval = X_eval[:, 1:].T

    X_test = X_test[:, :-1].T
    X_eval = X_eval[:, :-1].T

    model = SequenceProcessor(
        dim_in=1,
        dim_hidden=32,
        dim_out=1,
        num_layers=2,
        cell_type=LSTM,
    )

    criterion = losses.MSELoss()
    optim = optimizers.Nadam(model.parameters, learning_rate=1e-1)

    batch_inds = np.arange(len(X_train))

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_train = total_loss_eval = 0.0
        it = 0

        np.random.shuffle(batch_inds)
        X_train = X_train[batch_inds, :]

        for start in tqdm.auto.tqdm(np.arange(0, len(X_train), batch_size)):
            end = start + batch_size

            y_batch = X_train[start:end, 1:].T
            X_batch = X_train[start:end, :-1].T

            model.train()
            y_preds = model(X_batch)
            loss, loss_grad = criterion(y_batch, y_preds)
            model.backward(loss_grad)
            total_loss_train += loss

            optim.clip_grads_val()
            optim.step()

            model.eval()
            y_preds = model(X_eval)
            loss, _ = criterion(y_eval, y_preds)
            total_loss_eval += loss

            it += 1

        total_loss_train /= it
        total_loss_eval /= it

        print(f"Total loss (train) : {total_loss_train:.3f}")
        print(f"Total loss (eval)  : {total_loss_eval:.3f}")

    model.eval()
    y_preds = np.squeeze(model(X_test))
    test_rmse = float(np.sqrt(np.mean(np.square(y_preds - y_test))))
    print(f"Test RMSE: {test_rmse:.3f}")


if __name__ == "__main__":
    _test_nlp()
    _test_forecasting()

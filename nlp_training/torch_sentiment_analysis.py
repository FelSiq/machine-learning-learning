import typing as t

import numpy as np
import torch
import torch.nn as nn
import torchtext
import tqdm

import aclimdb_dataset_utils


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        rnn_hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super(Model, self).__init__()

        self._num_directions = 1 + int(bidirectional)
        self._num_layers = num_layers
        self._rnn_hidden_size = rnn_hidden_size
        self._dense_size = self._num_directions * rnn_hidden_size

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim,
            rnn_hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.dense = nn.Linear(self._dense_size, 1)

    def forward(self, X, X_orig_lens: t.Sequence[int]):
        out = self.embed(X)

        # Shape: (num_layers * num_directions, batch, hidden_size)
        _, out = self.rnn(out)

        if isinstance(self.rnn, nn.LSTM):
            # Note: LSTM outputs both hidden state cell and cell states
            # GRUs outputs only a single tensor.
            out = out[0]

        # Shape: (num_layers, num_directions, batch, hidden_size)
        out = out.view(
            self._num_layers, self._num_directions, -1, self._rnn_hidden_size
        )

        # Shape: (num_directions, batch, hidden_size)
        out = out[-1]

        # Shape: (batch, num_directions * hidden_size)
        if self._num_directions > 1:
            out = out.permute(1, 0, 2).reshape(-1, self._dense_size)

        else:
            out = out.squeeze()

        out = self.dropout(out)
        out = self.dense(out)
        return out


def predict(model, X, X_lens):
    y_preds = model(X, X_lens).detach().cpu().numpy()
    y_preds = (y_preds > 0.0).astype(int, copy=False)
    return y_preds.ravel()


def eval_model(
    model: nn.Module,
    data_gen: torch.utils.data.DataLoader,
) -> float:
    model.eval()

    eval_acc = 0
    total = 0

    for X_batch, y_batch, X_orig_lens in tqdm.auto.tqdm(data_gen):
        y_preds = predict(model, X_batch, X_orig_lens)
        y_batch = y_batch.cpu().numpy().astype(int)

        eval_acc += np.sum(y_preds == y_batch)

        total += len(y_batch)

    return eval_acc / total


def _test():
    device = "cuda"
    train_epochs = 5
    checkpoint_path = "model_torch_sentiment_analysis.pt"

    gen_train, gen_eval, vocab = aclimdb_dataset_utils.get_data(device)

    model = Model(
        vocab_size=len(vocab),
        embed_dim=32,
        rnn_hidden_size=64,
        num_layers=1,
        dropout=0.4,
        bidirectional=True,
    )

    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded checkpoint.")

    except FileNotFoundError:
        pass

    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    for i in np.arange(1, 1 + train_epochs):
        print(f"Epoch {i} / {train_epochs} ...")

        train_acc = 0.0
        train_size = 0

        model.train()

        for X_batch, y_batch, X_orig_lens in tqdm.auto.tqdm(gen_train):
            optim.zero_grad()
            y_preds = model(X_batch, X_orig_lens).squeeze()
            loss = criterion(y_preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optim.step()

            with torch.no_grad():
                train_acc += ((y_preds > 0.0) == y_batch).cpu().numpy().sum()

            train_size += len(X_batch)

        train_acc = train_acc / train_size
        eval_acc = eval_model(model, gen_eval)

        print(f"Train accuracy      : {train_acc:.4f}")
        print(f"Evaluation accuracy : {eval_acc:.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint in file {checkpoint_path}.")


if __name__ == "__main__":
    _test()

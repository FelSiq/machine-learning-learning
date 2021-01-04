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
        bidirectional: bool = False,
    ):
        super(Model, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            rnn_hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dense = nn.Linear(rnn_hidden_size, 1)

    def forward(self, X, X_orig_lens: t.Sequence[int]):
        out = self.embed(X)
        _, (out, _) = self.lstm(out)
        out = out.squeeze()
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

    gen_train, gen_eval, vocab = aclimdb_dataset_utils.get_data(
        device, max_len_train=None, max_len_eval=None
    )

    model = Model(
        vocab_size=len(vocab),
        embed_dim=64,
        rnn_hidden_size=128,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
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

import typing as t

import torch.nn as nn
import torch
import pandas as pd
import bpemb
import tqdm


class Model(nn.Module):
    def __init__(
        self,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        embeddings: torch.Tensor,
        padding_idx: int,
        dropout: float = 0.3,
    ):
        super(Model, self).__init__()

        num_emb_dim = embeddings.shape[1]

        _encoder_layers = nn.TransformerEncoderLayer(
            num_emb_dim,
            nhead,
            dim_feedforward,
            dropout=dropout,
        )

        self.weights = nn.Sequential(
            nn.Embedding.from_pretrained(embeddings, padding_idx=padding_idx),
            nn.TransformerEncoder(_encoder_layers, num_layers),
            nn.Linear(num_emb_dim, 1),
        )

    def forward(self, X):
        return self.weights(X).squeeze()[-1, :]


def get_data(
    max_input_len: int = 64, vocab_size: int = 25000, n: t.Optional[int] = None
):
    col_types = {"sentence": str, "label": int}
    data_train = pd.read_csv(
        "./corpus/SST-2/train.tsv", sep="\t", dtype=col_types, nrows=n
    )
    data_eval = pd.read_csv(
        "./corpus/SST-2/dev.tsv", sep="\t", dtype=col_types, nrows=n
    )

    codec = bpemb.BPEmb(lang="en", vs=vocab_size, add_pad_emb=True)
    pad_id = codec.spm.eos_id()

    def tokenize(data, labels):
        X = []
        y = []

        for inst, label in tqdm.auto.tqdm(zip(data, labels), total=len(labels)):
            inst = codec.encode_ids_with_eos(inst)

            if len(inst) <= max_input_len:
                inst += (max_input_len - len(inst)) * [pad_id]
                X.append(inst)
                y.append(label)

        return X, y

    X_train, y_train = tokenize(*data_train.values.T)
    X_eval, y_eval = tokenize(*data_eval.values.T)

    X_train = torch.tensor(X_train, dtype=torch.long)
    X_eval = torch.tensor(X_eval, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32)

    torch_data_train = torch.utils.data.TensorDataset(X_train, y_train)
    torch_data_eval = torch.utils.data.TensorDataset(X_eval, y_eval)

    del data_train, data_eval, col_types, pad_id

    return torch_data_train, torch_data_eval, codec


def _test():
    train_epochs = 10
    batch_size_train = 64
    batch_size_eval = 64
    max_input_len = 64
    device = "cuda"

    torch_data_train, torch_data_eval, codec = get_data(max_input_len=max_input_len)

    train_dataloader = torch.utils.data.DataLoader(
        torch_data_train,
        batch_size=batch_size_train,
        shuffle=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        torch_data_eval, batch_size=batch_size_eval
    )

    embeddings = torch.tensor(codec.vectors, dtype=torch.float32)

    model = Model(
        nhead=10,
        num_layers=2,
        dim_feedforward=128,
        embeddings=embeddings,
        padding_idx=codec.spm.eos_id(),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optim = torch.optim.Adam(model.parameters(), 1e-3)

    for epoch in range(1, 1 + train_epochs):
        print(f"Epoch: {epoch:4d} / {train_epochs:4d} ...")

        model.train()
        total_batches = 0
        train_loss = train_acc = 0.0

        for X_batch, y_batch in tqdm.auto.tqdm(train_dataloader):
            optim.zero_grad()

            X_batch = torch.transpose(X_batch, 0, 1)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_preds = model(X_batch)

            loss = criterion(y_preds, y_batch)
            loss.backward()
            optim.step()

            with torch.no_grad():
                total_batches += 1
                train_loss += loss.item()
                train_acc += (
                    ((y_preds > 0.5).long() == y_batch.long()).float().mean().item()
                )

        train_loss /= total_batches
        train_acc /= total_batches

        model.eval()
        total_batches = 0
        eval_loss = eval_acc = 0.0

        for X_batch, y_batch in tqdm.auto.tqdm(eval_dataloader):
            X_batch = torch.transpose(X_batch, 0, 1)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_preds = model(X_batch)
            loss = criterion(y_preds, y_batch)

            total_batches += 1
            eval_loss += loss.item()
            eval_acc += ((y_preds > 0.5).long() == y_batch.long()).float().mean().item()

        eval_loss /= total_batches
        eval_acc /= total_batches

        print(f"train loss: {train_loss:4.4f} - train acc: {train_acc:4.4f}")
        print(f"eval  loss: {eval_loss:4.4f} - eval  acc: {eval_acc:4.4f}")


if __name__ == "__main__":
    _test()

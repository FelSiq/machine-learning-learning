import typing as t

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import bpemb
import tqdm


class Model(nn.Module):
    def __init__(
        self,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        pretrained_emb: torch.Tensor,
        dropout: float = 0.0,
        freeze_emb: bool = False,
    ):
        super(Model, self).__init__()

        vocab_size, emb_dim = pretrained_emb.shape
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze_emb)
        self.rnn = nn.LSTM(
            emb_dim,
            rnn_hidden_dim,
            num_layers=rnn_num_layers,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, X, X_lens=None, hc_in=None, return_hc: bool = False):
        out = self.embedding(X)

        if X_lens is not None:
            out = nn.utils.rnn.pack_padded_sequence(out, X_lens, enforce_sorted=False)

        out, hc_out = self.rnn(out, hc_in)

        if X_lens is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out)

        out = self.dropout(out)
        out = self.linear(out)

        if return_hc:
            return out, hc_out

        return out


def train_step(model, criterion, optim, train_dataloader, codec, device):
    model.train()
    train_total_epochs = 0
    train_acc = train_loss = 0.0
    pad_id = codec.BOS

    for X_batch, X_lens in tqdm.auto.tqdm(train_dataloader):
        X_batch = torch.transpose(X_batch, 0, 1)
        X_batch = X_batch.to(device)

        optim.zero_grad()

        y_preds = model(X_batch, X_lens)
        y_preds = y_preds[:-1].reshape(-1, codec.vocab_size)
        y_true = X_batch[1:].reshape(-1).long()

        loss = criterion(y_preds, y_true)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optim.step()

        train_total_epochs += 1

        with torch.no_grad():
            train_loss += loss.item()
            mask = y_true != pad_id
            train_acc += (
                torch.masked_select(y_preds.argmax(dim=-1) == y_true, mask)
                .float()
                .mean()
                .item()
            )

    train_acc /= train_total_epochs
    train_loss /= train_total_epochs

    return train_loss, train_acc


def eval_step(model, criterion, eval_dataloader, codec, device):
    model.eval()
    eval_total_epochs = 0
    eval_acc = eval_loss = 0.0
    pad_id = codec.BOS

    for X_batch, X_lens in tqdm.auto.tqdm(eval_dataloader):
        X_batch = torch.transpose(X_batch, 0, 1)
        X_batch = X_batch.to(device)

        y_preds = model(X_batch, X_lens)
        y_preds = y_preds[:-1].reshape(-1, codec.vocab_size)
        y_true = X_batch[1:].reshape(-1).long()

        loss = criterion(y_preds, y_true)

        eval_total_epochs += 1

        eval_loss += loss.item()
        mask = y_true != pad_id
        eval_acc += (
            torch.masked_select(y_preds.argmax(dim=-1) == y_true, mask)
            .float()
            .mean()
            .item()
        )

    if eval_total_epochs:
        eval_acc /= eval_total_epochs
        eval_loss /= eval_total_epochs

    return eval_loss, eval_acc


def get_data(codec, max_length: int, train_frac: float = 0.95, verbose: bool = True):
    with open("../corpus/shakespeare.txt") as f:
        raw_data = f.readlines()

    data = list(
        map(
            lambda item: torch.tensor(
                codec.encode_ids_with_bos_eos(item.strip()), dtype=torch.long
            ),
            raw_data,
        )
    )

    train_size = int(train_frac * len(raw_data))

    train_dataset = data[:train_size]
    eval_dataset = data[train_size:]

    del raw_data, data

    if verbose:
        print("Train size   :", len(train_dataset))
        print("EVal size    :", len(eval_dataset))
        print("Train sample :", train_dataset[0])
        print("Eval sample  :", eval_dataset[0] if eval_dataset else None)

    return train_dataset, eval_dataset


def logsoftmax_sample(logits, temperature: float):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    u = torch.rand_like(log_probs) * (1 - 2e-6) + 1e-6
    g = -torch.log(-torch.log(u))
    return torch.argmax(log_probs + g * temperature, dim=-1)


def generate(model, max_length, codec, device, temperature: float):
    next_symbol = torch.tensor([codec.BOS], device=device, dtype=torch.long)
    next_symbol = next_symbol.unsqueeze(0)
    hc = None
    out = []

    for i in range(max_length):
        logits, hc = model(next_symbol, hc_in=hc, return_hc=True)
        next_symbol = logsoftmax_sample(logits, temperature)

        if next_symbol == codec.EOS:
            break

        out.append(next_symbol.item())

    return codec.decode_ids(out)


def _test():
    train_epochs = 80
    device = "cuda"
    max_length = 128
    train_batch_size = 64
    eval_batch_size = 128
    vocab_size = 10000
    emb_dim = 100
    lr_gamma = 0.9
    checkpoint_path = "lm_shakespeare_checkpoint.tar"

    rnn_hidden_dim = 256
    rnn_num_layers = 2
    dropout = 0.4

    codec = bpemb.BPEmb(lang="en", vs=vocab_size, dim=emb_dim)
    pretrained_emb = torch.tensor(codec.vectors)
    pad_id = codec.BOS

    def collate_fn(batch):
        X_lens = list(map(len, batch))
        X = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
        return X, X_lens

    train_dataset, eval_dataset = get_data(codec, max_length)

    model = Model(
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_num_layers=rnn_num_layers,
        pretrained_emb=pretrained_emb,
        dropout=dropout,
    )

    optim = torch.optim.Adam(model.parameters(), 1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=lr_gamma, patience=5, verbose=True
    )

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Checkpoint loaded.")

    except FileNotFoundError:
        pass

    model = model.to(device)

    if train_epochs > 0:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            collate_fn=collate_fn,
        )

        for epoch in range(1, 1 + train_epochs):
            print(f"Epoch: {epoch:4d} / {train_epochs:4d} ...")
            train_loss, train_acc = train_step(
                model, criterion, optim, train_dataloader, codec, device
            )
            eval_loss, eval_acc = eval_step(
                model, criterion, eval_dataloader, codec, device
            )
            # Note: I'm aware that i'm using the train loss instead of the eval
            # loss here, but this is what I actually care in this application
            # specifically.
            scheduler.step(train_loss)

            print(f"train loss: {train_loss:.4f} - train acc: {train_acc:.4f}")
            print(f"eval  loss: {eval_loss:.4f} - eval  acc: {eval_acc:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print("Done.")

    # Test model.
    print(40 * "=")
    temperature = 0.8

    for i in range(10):
        gen = generate(model, max_length, codec, device, temperature)
        print(f"Generated {i}:", gen)

    print(40 * "=")


if __name__ == "__main__":
    _test()

import typing as t

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import tqdm


class StarNameGenerator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, dropout: float):
        super(StarNameGenerator, self).__init__()

        self.rnn = nn.GRU(
            1,
            d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, X, X_lens):
        # Note: add embedding dimension
        X = X.unsqueeze(-1)

        out = nn.utils.rnn.pack_padded_sequence(X, X_lens, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        out = self.dropout(out)
        out = self.linear(out)

        return out


def get_data(max_length: int, train_frac: float = 0.9, verbose: bool = True):
    def process_word(word):
        seq = list(map(lambda item: vocab.get(item, vocab["<unk>"]), word))
        seq.append(vocab["<eos>"])
        seq = seq[:max_length]
        return torch.tensor(seq, dtype=torch.float32)

    data = pd.read_csv("./corpus/star_names.txt", squeeze=False, header=None)
    data = data.sample(frac=1, random_state=16)
    train_size = int(data.shape[0] * train_frac)

    unique_tokens = sorted(
        set.union(*data.iloc[:train_size, :].applymap(set).values.ravel())
    )
    unique_tokens = ["<pad>", "<unk>", "<eos>", "<bos>"] + unique_tokens

    vocab = dict(zip(unique_tokens, range(len(unique_tokens))))

    data = data.applymap(process_word).values.ravel().tolist()

    train_data = data[:train_size]
    eval_data = data[train_size:]

    if verbose:
        print("Train size   :", len(train_data))
        print("Eval size    :", len(eval_data))
        print("Vocab size    :", len(vocab))
        print("Vocab         :", vocab)
        print()
        print("Train example :", train_data[0])
        print("Eval example  :", eval_data[0])
        print()

    return train_data, eval_data, vocab


def train_step(model, criterion, optim, train_dataloader, vocab, device):
    model.train()
    train_loss = train_acc = 0.0
    train_total_batches = 0

    for X_batch, X_lens in tqdm.auto.tqdm(train_dataloader):
        optim.zero_grad()
        X_batch = torch.transpose(X_batch, 0, 1)
        X_batch = X_batch.to(device)
        y_preds = model(X_batch, X_lens)

        y_preds = y_preds[:-1].view(-1, len(vocab))
        y_true = X_batch[1:].reshape(-1).long()

        # Note: using teacher forcing to speed up training
        loss = criterion(y_preds, y_true)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        train_total_batches += 1

        with torch.no_grad():
            train_loss += loss.item()
            mask = y_true != vocab["<pad>"]
            train_acc += (
                torch.masked_select(y_preds.argmax(axis=-1) == y_true, mask)
                .float()
                .mean()
                .item()
            )

    train_loss /= train_total_batches
    train_acc /= train_total_batches

    return train_loss, train_acc


def eval_step(model, criterion, eval_dataloader, vocab, device):
    model.eval()
    eval_loss = eval_acc = 0.0
    eval_total_batches = 0

    for X_batch, X_lens in tqdm.auto.tqdm(eval_dataloader):
        X_batch = torch.transpose(X_batch, 0, 1)
        X_batch = X_batch.to(device)
        y_preds = model(X_batch, X_lens)

        y_preds = y_preds[:-1].view(-1, len(vocab))
        y_true = X_batch[1:].reshape(-1).long()

        loss = criterion(y_preds, y_true)

        eval_total_batches += 1

        eval_loss += loss.item()
        mask = y_true != vocab["<pad>"]
        eval_acc += (
            torch.masked_select(y_preds.argmax(axis=-1) == y_true, mask)
            .float()
            .mean()
            .item()
        )

    eval_loss /= eval_total_batches
    eval_acc /= eval_total_batches

    return eval_loss, eval_acc


def _test():
    train_epochs = 1000
    train_batch_size = 128
    eval_batch_size = 128
    max_length = 48
    device = "cuda"
    checkpoint_path = "lm_star_names.tar"

    train_dataset, eval_dataset, vocab = get_data(max_length)

    def collate_fn(batch):
        X_lens = list(map(len, batch))
        X = nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=vocab["<pad>"]
        )
        return X, X_lens

    model = StarNameGenerator(
        vocab_size=len(vocab), d_model=256, num_layers=1, dropout=0.4
    )

    optim = torch.optim.RMSprop(model.parameters(), 5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=20, factor=0.95, verbose=True
    )

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded checkpoint.")

    except FileNotFoundError:
        pass

    model = model.to(device)

    if train_epochs > 0:
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

        criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

        for epoch in range(1, 1 + train_epochs):
            print(f"Epoch: {epoch:4d} / {train_epochs:4d} ...")
            train_loss, train_acc = train_step(
                model, criterion, optim, train_dataloader, vocab, device
            )
            eval_loss, eval_acc = eval_step(
                model, criterion, eval_dataloader, vocab, device
            )
            scheduler.step(eval_loss)
            print(f"train loss: {train_loss:.4f} - train acc: {train_acc:.4f}")
            print(f"eval loss: {eval_loss:.4f} - eval acc: {eval_acc:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        print("Done.")

    # Test model.
    # print(40 * "=")
    # for i in range(10):
    #     print("Test:", i + 1)
    #     predict()


if __name__ == "__main__":
    _test()

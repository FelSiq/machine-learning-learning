import typing as t

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import tqdm


class StarNameGenerator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, dropout: float):
        super(StarNameGenerator, self).__init__()

        self.rnn = nn.LSTM(
            1,
            d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, X, X_lens=None, hc_in=None, return_hc: bool = False):
        # Note: add embedding dimension
        out = X.unsqueeze(-1)

        if X_lens is not None:
            out = nn.utils.rnn.pack_padded_sequence(out, X_lens, enforce_sorted=False)

        out, out_hc = self.rnn(out, hc_in)

        if X_lens is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out)

        out = self.dropout(out)
        out = self.linear(out)

        if return_hc:
            return out, out_hc

        return out


def get_data(max_length: int, train_frac: float = 0.9, verbose: bool = True):
    def process_word(word):
        seq = list(map(lambda item: vocab.get(item, vocab["<unk>"]), word))
        seq.append(vocab["<eos>"])
        seq = seq[:max_length]
        return torch.tensor(seq, dtype=torch.float32)

    data = pd.read_csv("../corpus/star_names.txt", squeeze=False, header=None)
    data = data.sample(frac=1, random_state=16)
    train_size = int(data.shape[0] * train_frac)

    unique_tokens = sorted(
        set.union(*data.iloc[:train_size, :].applymap(set).values.ravel())
    )
    unique_tokens = ["<pad>", "<unk>", "<eos>", "<bos>"] + unique_tokens

    vocab = dict(zip(unique_tokens, range(len(unique_tokens))))
    vocab_inv = dict(zip(vocab.values(), vocab.keys()))

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
        print("Eval example  :", eval_data[0] if eval_data else None)
        print()

    assert len(vocab) == len(vocab_inv)

    return train_data, eval_data, vocab, vocab_inv


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

    if eval_total_batches:
        eval_loss /= eval_total_batches
        eval_acc /= eval_total_batches

    return eval_loss, eval_acc


def logsoftmax_sample(logits, temperature=1.0):
    assert 0 <= temperature <= 1.0

    log_probs = nn.functional.log_softmax(logits, dim=-1)
    # Note: sample uniform U(1e-6, 1 - 1e-6)
    u = torch.rand_like(log_probs) * (1 - 2e-6) + 1e-6
    g = -torch.log(-torch.log(u))
    return torch.argmax(log_probs + g * temperature, axis=-1)


def generate(
    model,
    max_length: int,
    vocab,
    vocab_inv,
    device,
    temperature: float = 1.0,
    start_id: int = None,
):
    model.eval()
    hc = None

    if start_id is None:
        start_id = np.random.choice(
            [vocab.get(chr(c), vocab["A"]) for c in range(ord("A"), 1 + ord("Z"))]
        )

    out = [vocab_inv[start_id]]

    next_token = torch.tensor([start_id], dtype=torch.float)
    next_token = next_token.unsqueeze(0)
    next_token = next_token.to(device)

    for i in range(1, 1 + max_length):
        next_token = next_token.float()
        logits, hc = model(next_token, hc_in=hc, return_hc=True)
        next_token = logsoftmax_sample(logits, temperature)

        if next_token == vocab["<eos>"]:
            break

        out.append(vocab_inv[int(next_token.item())])

    return "".join(out)


def _test():
    train_epochs = 0
    train_batch_size = 128
    eval_batch_size = 128
    max_length = 48
    device = "cuda"
    checkpoint_path = "lm_star_names.tar"

    train_dataset, eval_dataset, vocab, vocab_inv = get_data(max_length, train_frac=1.0)

    def collate_fn(batch):
        X_lens = list(map(len, batch))
        X = nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=vocab["<pad>"]
        )
        return X, X_lens

    model = StarNameGenerator(
        vocab_size=len(vocab), d_model=256, num_layers=3, dropout=0.5
    )

    optim = torch.optim.RMSprop(model.parameters(), 5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=40,
        gamma=0.95,
        verbose=True,
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
            scheduler.step()
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
    print(40 * "=")

    raw_data = frozenset(
        pd.read_csv("../corpus/star_names.txt", squeeze=True, header=None)
    )
    memorized = []
    is_in_train_set = 0

    for i in range(20):
        res = generate(model, max_length, vocab, vocab_inv, device, temperature=0.8)
        print("Test:", i + 1, res)
        if res in raw_data:
            is_in_train_set += 1
            memorized.append(res)

    print("Generated instances memorized in network:", is_in_train_set)
    print("Memorized:", memorized)
    print(40 * "=")


if __name__ == "__main__":
    _test()

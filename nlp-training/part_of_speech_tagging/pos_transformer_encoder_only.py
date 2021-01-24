import typing as t
import math

import torch.nn as nn
import torch
import tqdm

import pos_data_utils


class PositionalEncoding(nn.Module):
    # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.0, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class POSTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_num: int,
        max_len: int,
        emb_dim: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super(POSTagger, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = PositionalEncoding(emb_dim, dropout, max_len)
        self.weights = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, tag_num),
        )

    def forward(self, X):
        out = self.emb(X)
        out = self.pos_emb(out)
        out = self.weights(out)
        return out


def train_step(model, criterion, optim, train_dataloader, vocab_tags, device):
    train_acc = train_loss = 0.0
    train_total_batches = 0

    for X_batch, y_batch in tqdm.auto.tqdm(train_dataloader):
        X_batch = torch.transpose(X_batch, 0, 1)
        y_batch = torch.transpose(y_batch, 0, 1)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optim.zero_grad()

        y_preds = model(X_batch)

        y_preds = y_preds[:-1].reshape(-1, len(vocab_tags))
        y_batch = y_batch[1:].reshape(-1)

        loss = criterion(y_preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optim.step()

        train_total_batches += 1

        with torch.no_grad():
            train_loss += loss.item()
            mask = y_batch != vocab_tags["<pad>"]
            train_acc += (
                torch.masked_select(y_preds.argmax(dim=-1) == y_batch, mask)
                .float()
                .mean()
                .item()
            )

    train_loss /= train_total_batches
    train_acc /= train_total_batches

    return train_loss, train_acc


def eval_step(model, criterion, eval_dataloader, vocab_tags, device):
    eval_acc = eval_loss = 0.0
    eval_total_batches = 0

    for X_batch, y_batch in tqdm.auto.tqdm(eval_dataloader):
        X_batch = torch.transpose(X_batch, 0, 1)
        y_batch = torch.transpose(y_batch, 0, 1)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_preds = model(X_batch)

        y_preds = y_preds[:-1].reshape(-1, len(vocab_tags))
        y_batch = y_batch[1:].reshape(-1)

        loss = criterion(y_preds, y_batch)

        eval_total_batches += 1

        eval_loss += loss.item()
        mask = y_batch != vocab_tags["<pad>"]
        eval_acc += (
            torch.masked_select(y_preds.argmax(dim=-1) == y_batch, mask)
            .float()
            .mean()
            .item()
        )

    eval_loss /= eval_total_batches
    eval_acc /= eval_total_batches

    return eval_loss, eval_acc


def _test():
    train_epochs = 3
    max_len = 64
    train_batch_size = 32
    eval_batch_size = 32
    device = "cuda"
    lr_gamma = 0.95
    lr_step_size = 1
    checkpoint_path = "pos_enconly_checkpoint.tar"

    (
        train_dataset,
        eval_dataset,
        vocab_words,
        vocab_tags,
    ) = pos_data_utils.get_data(max_len=max_len)

    model = POSTagger(
        vocab_size=len(vocab_words),
        tag_num=len(vocab_tags),
        max_len=max_len,
        emb_dim=100,
        num_layers=8,
        nhead=10,
        dim_feedforward=256,
        dropout=0.1,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=eval_batch_size
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tags["<pad>"])
    optim = torch.optim.Adam(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=lr_step_size, gamma=lr_gamma
    )

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Checkpoint loaded.")

    except FileNotFoundError:
        pass

    model = model.to(device)

    for epoch in range(1, 1 + train_epochs):
        print(f"Epoch: {epoch:4d} / {train_epochs:4d} ...")
        train_loss, train_acc = train_step(
            model, criterion, optim, train_dataloader, vocab_tags, device
        )
        eval_loss, eval_acc = eval_step(
            model, criterion, eval_dataloader, vocab_tags, device
        )
        scheduler.step()
        print(f"train loss: {train_loss:.4f} - train acc: {train_acc:.4f}")
        print(f"eval  loss: {eval_loss:.4f} - eval  acc: {eval_acc:.4f}")

    checkpoint = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)

    print("Done.")


if __name__ == "__main__":
    _test()

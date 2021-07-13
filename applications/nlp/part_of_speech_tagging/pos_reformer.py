import typing as t
import math

import torch.nn as nn
import torch
import tqdm
import reformer_pytorch

import pos_data_utils


class POSTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_num: int,
        max_len: int,
        axial_position_shape: t.Tuple[int, int],
        emb_dim: int,
        num_layers: int,
        nhead: int,
        bucket_size: int,
        dim_feedforward: int,
        ff_chunks: int, 
        dropout: float,
    ):
        super(POSTagger, self).__init__()

        self.encoder = reformer_pytorch.ReformerLM(
            num_tokens=vocab_size,
            emb_dim=emb_dim,
            dim=dim_feedforward,
            depth=num_layers,
            heads=nhead,
            lsh_dropout=dropout,
            ff_dropout=dropout,
            post_attn_dropout=dropout,
            layer_dropout=dropout,
            bucket_size=bucket_size,
            max_seq_len=max_len,
            ff_chunks=ff_chunks,
            axial_position_shape=axial_position_shape,
            return_embeddings=True,
        )

        self.linear = nn.Linear(emb_dim, tag_num)


    def forward(self, X):
        out = self.encoder(X)
        out = self.linear(out)
        return out


def train_step(model, criterion, optim, train_dataloader, vocab_tags, device):
    train_acc = train_loss = 0.0
    train_total_batches = 0

    for X_batch, y_batch in tqdm.auto.tqdm(train_dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optim.zero_grad()

        y_preds = model(X_batch, y_batch)

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
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_preds = model(X_batch, y_batch)

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
    train_batch_size = 256
    eval_batch_size = 256
    device = "cuda"
    lr_gamma = 0.95
    lr_step_size = 1
    checkpoint_path = "pos_reformer_checkpoint.tar"

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
        emb_dim=32,
        num_layers=8,
        nhead=8,
        bucket_size=32,
        ff_chunks=10,
        dim_feedforward=32,
        axial_position_shape=(16, 8),  # Note: must multiply to max_seq_len
        dropout=0.1,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=eval_batch_size
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tags["<pad>"])
    optim = torch.optim.Adam(model.parameters(), 3e-4)
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

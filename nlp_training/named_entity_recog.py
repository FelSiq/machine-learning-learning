import typing as t
import collections
import functools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import ner_utils


class RNNNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_tags: int,
        num_lstm: int = 1,
    ):
        super(RNNNER, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.pack_func = functools.partial(
            nn.utils.rnn.pack_padded_sequence, batch_first=True, enforce_sorted=False
        )

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_lstm,
            bidirectional=True,
            batch_first=True,
        )

        self.pad_func = functools.partial(
            nn.utils.rnn.pad_packed_sequence, batch_first=True
        )

        self.logits = nn.Linear(2 * embed_dim, num_tags)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, X_lens):
        out = self.embedding(X)
        out = self.pack_func(out, X_lens)
        out = self.rnn(out)[0]
        out = self.pad_func(out)[0]
        out = self.logits(out)
        out = self.softmax(out)
        return out


def collate_pad_seqs(X_batch, y_batch, vocab, tags):
    X_orig_lens = list(map(len, X_batch))
    y_orig_lens = list(map(len, y_batch))

    X_batch_padded = nn.utils.rnn.pad_sequence(
        X_batch, batch_first=True, padding_value=vocab["<PAD>"]
    )
    y_batch_padded = nn.utils.rnn.pad_sequence(
        y_batch, batch_first=True, padding_value=tags["O"]
    )

    return X_batch_padded, y_batch_padded, X_orig_lens, y_orig_lens


def data_gen(X, y, batch_size: int, vocab, tags):
    inds = np.arange(len(X))
    np.random.shuffle(inds)

    for i in np.arange(0, inds.size, batch_size):
        X_batch = []
        y_batch = []

        for j in inds[i : i + batch_size]:
            X_batch.append(X[j])
            y_batch.append(y[j])

        yield collate_pad_seqs(X_batch, y_batch, vocab, tags)


def masked_cross_entropy(y_preds, y_batch, y_len, device):
    loss = -torch.log(torch.gather(y_preds, 2, y_batch.unsqueeze(2)).squeeze())
    mask = torch.arange(y_preds.shape[1]).repeat((y_preds.shape[0], 1)).to(device)

    for i, l in enumerate(y_len):
        mask[i] = mask[i] < l

    loss = loss.masked_select(mask.bool()).mean()

    return loss


def _test():
    min_freq_vocab = 2
    hold_out_validation_frac = 0.025
    train_epochs = 2
    batch_size = 512
    embed_dim = 128
    num_lstm = 1
    device = "cuda"

    X, y, vocab, tags = ner_utils.get_data(min_freq_vocab)

    ner_utils.token_to_ind(X, vocab, unk_ind=vocab["<UNK>"])
    print("Processed X: tokens -> indices")
    ner_utils.token_to_ind(y, tags)
    print("Processed y: tokens -> indices")

    print("TAGS:")
    print(tags)
    print("Number of tags     :", len(tags))

    assert len(X) == len(y)

    print("Number of sentences:", len(X))
    print("Vocabulary size    :", len(vocab))

    for i in np.arange(len(X)):
        X[i] = torch.Tensor(X[i]).long()
        y[i] = torch.Tensor(y[i]).long()

    train_size = int(len(X) * (1.0 - hold_out_validation_frac))
    X_train, y_train = X[:train_size], y[:train_size]
    X_eval, y_eval = X[train_size:], y[train_size:]

    X_eval, _, X_eval_lens, _ = collate_pad_seqs(X_eval, y_eval, vocab=vocab, tags=tags)
    X_eval = X_eval.to(device)

    print("Train size:", train_size)
    print("Eval size :", len(X) - train_size)

    model = RNNNER(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_tags=len(tags),
        num_lstm=num_lstm,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in np.arange(train_epochs):
        print(f"epoch: {1 + epoch} / {train_epochs} ...")
        dataloader = data_gen(
            X_train, y_train, batch_size=batch_size, vocab=vocab, tags=tags
        )
        model.train()

        for b, (X_batch, y_batch, X_lens, y_lens) in enumerate(dataloader, 1):
            print(f"\r{b} / {(len(X_train) + batch_size - 1) // batch_size}", end=" ")
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optim.zero_grad()
            y_preds = model(X_batch, X_lens)
            loss = masked_cross_entropy(y_preds, y_batch, y_lens, device)
            loss.backward()
            optim.step()

        print("\nEvaluating...")
        model.eval()
        y_preds = model(X_eval, X_eval_lens).argmax(axis=2).detach().cpu()
        eval_acc = 0.0

        for k, y_cur in enumerate(y_eval):
            preds_cur = y_preds[k, : len(y_cur)]
            eval_acc += sum(y_cur == preds_cur).item() / len(y_cur)

        eval_acc /= len(y_eval)
        print(f"Evaluation acc: {eval_acc:.4f}")

    print("Done.")


if __name__ == "__main__":
    _test()

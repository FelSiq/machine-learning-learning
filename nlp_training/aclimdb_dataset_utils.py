import typing as t
import os
import pickle
import random

import torch
import torch.nn as nn
import nltk


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, X, y, device: str):
        self._device = torch.device(device)
        self.X = [torch.tensor(x, dtype=torch.long, device=self._device) for x in X]
        self.y = y
        self.seed = 16

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        random.seed(self.seed)
        random.shuffle(self.X)

        random.seed(self.seed)
        random.shuffle(self.y)

        self.seed += 1

        return iter(zip(self.X, self.y))

    def __len__(self):
        return len(self.X)


def get_vocab(base_dir_path: str, unk_token: str = "<UNK>", pad_token: str = "<PAD>"):
    vocab = dict()
    vocab[pad_token] = 0
    vocab[unk_token] = 1

    with open(os.path.join(base_dir_path, "imdb.vocab")) as f:
        unique_tokens = set(f.read().strip().split("\n"))

    vocab.update(
        zip(
            sorted(unique_tokens),
            range(1 + len(vocab), 1 + len(vocab) + len(unique_tokens)),
        )
    )

    return vocab


def load_dir(base_dir_path: str, train: bool):
    X, y = [], []

    for cls_ind, cls_name in enumerate(("neg", "pos")):
        dir = os.path.join(base_dir_path, "train" if train else "test", cls_name)
        for file in os.listdir(dir):
            with open(os.path.join(dir, file), "r") as f:
                text = f.read().strip().lower()
                tokens = nltk.tokenize.word_tokenize(text)
                X.append(tokens)

            y.append(cls_ind)

    return X, y


def preprocess_data(verbose: bool = True):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir_path = os.path.join(script_dir, "corpus/aclImdb")

    vocab = get_vocab(base_dir_path)

    X_train, y_train = load_dir(base_dir_path, train=True)
    X_eval, y_eval = load_dir(base_dir_path, train=False)

    if verbose:
        print(f"Got {len(vocab)} tokens in vocab file.")
        print(f"Train dataset size: {len(X_train)}.")
        print(f"Test dataset size : {len(X_eval)}.")

    for tokens in X_train:
        for j, token in enumerate(tokens):
            tokens[j] = vocab.get(token, vocab["<UNK>"])

    for tokens in X_eval:
        for j, token in enumerate(tokens):
            tokens[j] = vocab.get(token, vocab["<UNK>"])

    res = (X_train, y_train, X_eval, y_eval, vocab)

    out_path = os.path.join(script_dir, "corpus", "aclimdb_preprocessed.pickle")

    with open(out_path, "wb") as fout:
        pickle.dump(res, fout)

    print(f"Crated output .pickle file with preprocessed data: {out_path}")


def get_data(
    device: str,
    max_len_train: t.Optional[int] = None,
    max_len_eval: t.Optional[int] = None,
):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    with open(
        os.path.join(script_dir, "corpus", "aclimdb_preprocessed.pickle"), "rb"
    ) as fin:
        X_train, y_train, X_eval, y_eval, vocab = pickle.load(fin)

    random.seed(16)
    random.shuffle(X_train)
    random.shuffle(X_eval)

    random.seed(16)
    random.shuffle(y_train)
    random.shuffle(y_eval)

    if max_len_train is not None:
        X_train = X_train[:max_len_train]
        y_train = y_train[:max_len_train]

    if max_len_eval is not None:
        X_eval = X_eval[:max_len_eval]
        y_eval = y_eval[:max_len_eval]

    def collate_pad_seqs_fn(batch):
        X_batch, y_batch = zip(*batch)

        X_orig_lens = (torch.tensor(list(map(len, X_batch)), dtype=torch.long)).to(
            device
        )

        X_padded = nn.utils.rnn.pad_sequence(
            X_batch, batch_first=True, padding_value=vocab["<PAD>"]
        )

        y_batch = torch.tensor(y_batch, dtype=torch.float, device=device)

        return X_padded, y_batch, X_orig_lens

    train_dataset = IterableDataset(X_train, y_train, device)
    eval_dataset = IterableDataset(X_eval, y_eval, device)

    gen_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=collate_pad_seqs_fn,
    )

    gen_eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=128,
        collate_fn=collate_pad_seqs_fn,
    )

    return gen_train, gen_eval, vocab


if __name__ == "__main__":
    preprocess_data()

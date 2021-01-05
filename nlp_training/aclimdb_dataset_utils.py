import typing as t
import os
import pickle
import random
import collections

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
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            random.seed(self.seed)
            random.shuffle(self.X)

            random.seed(self.seed)
            random.shuffle(self.y)

            self.seed += 1

            return iter(zip(self.X, self.y))

        total_size = len(self.X)

        batch_per_worker = (
            total_size + worker_info.num_workers - 1
        ) // worker_info.num_workers

        ind_start = batch_per_worker * worker_info.id
        ind_end = min(ind_start + batch_per_worker, len(self.X))

        return iter(zip(self.X[ind_start:ind_end], self.y[ind_start:ind_end]))

    def __len__(self):
        return len(self.X)


def get_vocab(
    X: t.Sequence[t.Sequence[str]],
    stop_words: t.Set[str],
    unk_token: str = "<UNK>",
    pad_token: str = "<PAD>",
    min_freq_count: int = 2,
):
    freqs = collections.Counter()

    for x in X:
        freqs.update(map(str.lower, x))

    unique_tokens = sorted(
        [k for k, v in freqs.items() if v >= min_freq_count and k not in stop_words]
    )

    vocab = dict()
    vocab[pad_token] = 0
    vocab[unk_token] = 1

    vocab.update(
        zip(
            unique_tokens,
            range(len(vocab), len(vocab) + len(unique_tokens)),
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


def preprocess_sentence(
    tokens: t.Sequence[str], vocab, stop_words: t.Set[str]
) -> t.Sequence[str]:
    new_sentence = [
        vocab.get(token.lower(), vocab["<UNK>"])
        for token in tokens
        if token not in stop_words
    ]
    return new_sentence


def preprocess_sentences(
    X: t.Sequence[t.Sequence[str]], y: t.Sequence[t.Any], vocab, stop_words: t.Set[str]
):
    for i, tokens in enumerate(X):
        X[i] = preprocess_sentence(tokens, vocab, stop_words)

    # Note: remove possibly empty instances
    i = 0

    while i < len(X):
        if not X[i]:
            X.pop(i)
            y.pop(i)

        else:
            i += 1


def preprocess_data(
    remove_stop_words: bool = False,
    verbose: bool = True,
    mix_test_in_train_frac: float = 0.0,
):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir_path = os.path.join(script_dir, "corpus/aclImdb")

    X_train, y_train = load_dir(base_dir_path, train=True)
    X_eval, y_eval = load_dir(base_dir_path, train=False)

    random.seed(16)
    random.shuffle(X_train)
    random.shuffle(X_eval)

    random.seed(16)
    random.shuffle(y_train)
    random.shuffle(y_eval)

    if mix_test_in_train_frac > 0.0:
        size = int(mix_test_in_train_frac * len(X_eval))

        X_train.extend(X_eval[:size])
        y_train.extend(y_eval[:size])

        X_eval = X_eval[size:]
        y_eval = y_eval[size:]

    if remove_stop_words:
        stop_words = set(nltk.corpus.stopwords.words("english"))
        if verbose:
            print("Will remove stop words.")

    else:
        stop_words = set()

    vocab = get_vocab(X_train, stop_words)

    if verbose:
        print(f"Got {len(vocab)} tokens in vocab file.")
        print(f"Train dataset size : {len(X_train)}.")
        print(f"Test dataset size  : {len(X_eval)}.")

    preprocess_sentences(X_train, y_train, vocab, stop_words)
    preprocess_sentences(X_eval, y_eval, vocab, stop_words)

    if verbose:
        print(f"Train dataset size after preprocessing : {len(X_train)}.")
        print(f"Test dataset size after preprocessing  : {len(X_eval)}.")

    res = (X_train, y_train, X_eval, y_eval, vocab)

    out_path = os.path.join(script_dir, "corpus", "aclimdb_preprocessed.pickle")

    with open(out_path, "wb") as fout:
        pickle.dump(res, fout)

    print(f"Crated output .pickle file with preprocessed data: {out_path}")


def get_data(
    device: str,
    max_len_train: t.Optional[int] = None,
    max_len_eval: t.Optional[int] = None,
    verbose: bool = True,
):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    with open(
        os.path.join(script_dir, "corpus", "aclimdb_preprocessed.pickle"), "rb"
    ) as fin:
        X_train, y_train, X_eval, y_eval, vocab = pickle.load(fin)

    if max_len_train is not None:
        X_train = X_train[:max_len_train]
        y_train = y_train[:max_len_train]

    if max_len_eval is not None:
        X_eval = X_eval[:max_len_eval]
        y_eval = y_eval[:max_len_eval]

    if verbose:
        print("Train data size :", len(X_train))
        print("Eval data size  :", len(X_eval))

    assert len(X_train) == len(y_train)
    assert len(X_eval) == len(y_eval)

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
    nltk.download("stopwords")
    preprocess_data(remove_stop_words=True, mix_test_in_train_frac=0.75)

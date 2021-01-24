import typing as t
import functools

import pandas as pd
import numpy as np
import torch


def process_token(token, vocab):
    return vocab.get(token, vocab["<unk>"])


def token_to_ind(sentences, vocab):
    partial_process_token = functools.partial(process_token, vocab=vocab)
    for i, sentence in enumerate(sentences):
        sentences[i] = list(map(partial_process_token, sentence))


def get_data(max_len: int, train_frac: float = 0.99, verbose: bool = True):
    raw_data = pd.read_csv(
        "../corpus/ner_dataset.csv",
        encoding="ISO-8859-1",
        usecols=[0, 1, 2],
        names=["is_start", "words", "pos"],
        header=0,
    )
    sentences = []
    tags = []

    unique_words = set()
    unique_tags = set()

    bos_inds = np.hstack(
        (np.flatnonzero(raw_data["is_start"].notna()), raw_data.shape[0])
    )

    for ind_start, ind_end in zip(bos_inds[:-1], bos_inds[1:]):
        if ind_end - ind_start + 2 > max_len:
            continue

        inst = raw_data.iloc[ind_start:ind_end]

        new_sentence = inst["words"].values.tolist()
        new_tags = inst["pos"].values.tolist()

        new_sentence.append("<eos>")
        new_tags.append("<eos>")

        pad_len = max_len - len(new_sentence)
        new_sentence.extend(pad_len * ["<pad>"])
        new_tags.extend(pad_len * ["<pad>"])

        assert len(new_sentence) == len(new_tags)

        sentences.append(new_sentence)
        tags.append(new_tags)

    train_size = int(train_frac * len(sentences))

    train_sentences = sentences[:train_size]
    eval_sentences = sentences[train_size:]

    train_tags = tags[:train_size]
    eval_tags = tags[train_size:]

    for sentence, tags in zip(train_sentences, train_tags):
        # Note: skipping EOS/BOS TOKENS
        unique_words.update(sentence)
        unique_tags.update(tags)

    unique_words = ["<unk>"] + sorted(unique_words)
    unique_tags = ["<unk>"] + sorted(unique_tags)

    vocab_words = dict(zip(unique_words, range(len(unique_words))))
    vocab_tags = dict(zip(unique_tags, range(len(unique_tags))))

    if verbose:
        print("Train size      :", len(train_sentences))
        print("Eval size       :", len(eval_sentences))
        print("Vocab size      :", len(vocab_words))
        print("POS tags number :", len(vocab_tags))
        print()
        print("Sample train sentence:", train_sentences[0])
        print("Sample train POS tags:", train_tags[0])
        print("Sample eval sentence:", eval_sentences[0])
        print("Sample eval POS tags:", eval_tags[0])

    assert len(train_sentences) == len(train_tags)
    assert len(eval_sentences) == len(eval_tags)

    del unique_words, unique_tags, raw_data

    token_to_ind(train_sentences, vocab_words)
    token_to_ind(eval_sentences, vocab_words)
    token_to_ind(train_tags, vocab_tags)
    token_to_ind(eval_tags, vocab_tags)

    train_sentences = torch.tensor(train_sentences, dtype=torch.long)
    train_tags = torch.tensor(train_tags, dtype=torch.long)
    eval_sentences = torch.tensor(eval_sentences, dtype=torch.long)
    eval_tags = torch.tensor(eval_tags, dtype=torch.long)

    if verbose:
        print("Sample processed train sentence:", train_sentences[0])
        print("Sample processed train POS tags:", train_tags[0])
        print("Sample processed eval sentence:", eval_sentences[0])
        print("Sample processed eval POS tags:", eval_tags[0])

    train_dataset = torch.utils.data.TensorDataset(train_sentences, train_tags)
    eval_dataset = torch.utils.data.TensorDataset(eval_sentences, eval_tags)

    return train_dataset, eval_dataset, vocab_words, vocab_tags


def _test():
    get_data()


if __name__ == "__main__":
    _test()

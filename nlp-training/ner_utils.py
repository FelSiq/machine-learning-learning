"""Named Entity Recognition (NER) utilities."""
import collections
import pickle

import pandas as pd


def get_corpus(it_to_print: int = 1000):
    data = pd.read_csv(
        "corpus/ner_dataset.csv",
        encoding="ISO-8859-1",
        index_col=0,
        header=0,
    )

    X = []
    y = []

    cur_x = []
    cur_y = []

    for j, (i, (word, _, tag)) in enumerate(data.iterrows(), 1):
        if pd.isna(i):
            cur_x.append(word)
            cur_y.append(tag)

        else:
            if cur_x:
                X.append(cur_x)
                y.append(cur_y)

            cur_x = [word]
            cur_y = [tag]

        if it_to_print > 0 and j % it_to_print == 0:
            print(f"{j} / {data.shape[0]} ... ")

    if cur_x:
        X.append(cur_x)
        y.append(cur_y)

    tags = sorted(set(data["Tag"]))
    tags = dict(zip(tags, range(len(tags))))

    return X, y, tags


def get_vocab(
    X,
    min_freq: int = 1,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
    eos_token: str = "<EOS>",
):
    assert min_freq >= 1

    freqs = collections.Counter()

    for inst in X:
        freqs.update(inst)

    if min_freq > 1:
        freqs = {k: v for k, v in freqs.items() if v >= min_freq}

    vocab = {
        pad_token: 0,
        unk_token: 1,
        eos_token: 2,
    }

    vocab.update(dict(zip(sorted(freqs), range(len(vocab), len(vocab) + len(freqs)))))

    return vocab


def dump_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)

    return obj


def get_data(min_freq_vocab: int = 2):
    X = load_pickle("corpus/ner_X.pickle")
    y = load_pickle("corpus/ner_y.pickle")
    tags = load_pickle("corpus/ner_tags.pickle")
    vocab = load_pickle(f"corpus/ner_vocab_min_freq_{min_freq_vocab}.pickle")
    return X, y, vocab, tags


def token_to_ind(data, indices, unk_ind: int = -1):
    for row in data:
        for j, token in enumerate(row):
            row[j] = indices.get(token, unk_ind)


def _run():
    min_freq = 2

    X, y, tags = get_corpus()
    vocab = get_vocab(X, min_freq=2)

    dump_pickle(X, "corpus/ner_X.pickle")
    dump_pickle(y, "corpus/ner_y.pickle")
    dump_pickle(tags, "corpus/ner_tags.pickle")
    dump_pickle(vocab, f"corpus/ner_vocab_min_freq_{min_freq}.pickle")


if __name__ == "__main__":
    _run()

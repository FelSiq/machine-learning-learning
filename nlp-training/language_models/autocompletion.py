import typing as t
import re
import collections
import pickle

import numpy as np
import pandas as pd
import nltk


def log_perplexity(
    words: t.Union[t.Sequence[str], str],
    model: t.Dict[str, t.Any],
) -> float:
    if isinstance(words, str):
        words, _ = preproc_sentences([words], vocab=set(model["sorted_vocab"]))[0]

    start_token = model["start_token"]
    end_token = model["end_token"]
    len_vocab = len(model["sorted_vocab"])
    n = model["n"]
    m = len(words) + 1

    prob_mat = model["prob_mat"]

    words = (n - 1) * [start_token] + words + [end_token]
    log_p = 0.0

    for i in np.arange(n - 1, len(words)):
        n_1m_gram, word = tuple(words[i - n + 1 : i]), words[i]

        if n_1m_gram in prob_mat.index:
            p = prob_mat.loc[n_1m_gram, word]

        else:
            p = 1.0 / len_vocab

        log_p += np.log2(p)

    log_perp = -log_p / m

    return log_perp


def preproc_sentences(
    sentences: t.Sequence[str],
    min_vocab_freq: int = 2,
    unknown_token: str = "<UNK>",
    vocab: t.Optional[t.Set[str]] = None,
) -> t.Tuple[t.Sequence[t.Sequence[str]], t.Set[str]]:
    assert unknown_token
    assert min_vocab_freq >= 0

    clean_re = re.compile(r"[^a-z0-9.!?'<> ]+")

    preproc_sentences = []

    word_freqs = collections.Counter()

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = clean_re.sub("", sentence)
        words = nltk.word_tokenize(sentence)

        if vocab is None:
            word_freqs.update(words)

        preproc_sentences.append(words)

    if vocab is None:
        vocab = {k for k, v in word_freqs.items() if v >= min_vocab_freq}

    vocab.add(unknown_token)

    for sentence in preproc_sentences:
        for i in np.arange(len(sentence)):
            if sentence[i] not in vocab:
                sentence[i] = unknown_token

    return preproc_sentences, vocab


def build_n_gram(
    sentences: t.Sequence[t.Sequence[str]],
    vocab: t.Set[str],
    n: int,
    start_token: str = "<s>",
    end_token: str = "</s>",
    epsilon: t.Union[int, float] = 1,
) -> t.Dict[str, t.Any]:
    assert epsilon >= 0.0
    assert n >= 2

    sorted_vocab = sorted(vocab.union({start_token, end_token}))

    n_grams = collections.defaultdict(int)  # type: t.Dict[t.Tuple[str, ...], str]

    for words in sentences:
        words = (n - 1) * [start_token] + words + [end_token]
        for i in np.arange(n, len(words) + 1):
            word_seq = tuple(words[i - n : i])
            n_grams[word_seq] += 1

    index = sorted({k[:-1] for k in n_grams.keys()})

    prob_mat = pd.DataFrame(
        epsilon,
        index=pd.MultiIndex.from_tuples(index),
        columns=sorted_vocab,
        dtype=np.float32,
    )

    for word_seq, freq in n_grams.items():
        n_m1_gram, word = word_seq[:-1], word_seq[-1]
        prob_mat.loc[n_m1_gram, word] += freq

    prob_mat = prob_mat.div(prob_mat.sum(axis=1), axis=0)

    model = {
        "sorted_vocab": sorted_vocab,
        "prob_mat": prob_mat,
        "n": n,
        "start_token": start_token,
        "end_token": end_token,
        "epsilon": epsilon,
    }

    return model


def get_sentences(n: t.Optional[int] = None):
    with open("../corpus/en_US.twitter.txt") as f:
        corpus = f.read()

    sentences = corpus.split("\n")

    if n is not None:
        sentences = sentences[:n]

    sentences = [s.strip() for s in sentences if s]

    return sentences


def autocomplete(
    prev_tokens: t.Union[str, t.Sequence[str]],
    model: t.Dict[str, t.Any],
    suggestions_num: int = 3,
    random: bool = True,
):
    prob_mat = model["prob_mat"]
    n = model["n"]
    start_token = model["start_token"]

    prev_tokens = (n - 1) * [start_token] + prev_tokens
    prev_tokens = tuple(prev_tokens[-n + 1 :])

    suggestions = ["</s>"]

    if prev_tokens in prob_mat.index:
        if not random:
            inds = (-prob_mat.loc[prev_tokens, :].values).argsort()[:suggestions_num]

        else:
            inds = np.random.choice(
                prob_mat.columns.size,
                p=prob_mat.loc[prev_tokens, :].values,
                size=suggestions_num,
                replace=False,
            )

        suggestions = list(prob_mat.columns[inds])

    return suggestions


def _test():
    n = 4

    filename = f"n_gram_models/{n}_gram_model.pickle"

    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)

    except FileNotFoundError:
        import random

        random.seed(87)
        nltk.download("punkt")

        sentences = get_sentences()

        random.shuffle(sentences)

        train_size = int(len(sentences) * 0.95)
        train_data = sentences[:train_size]
        test_data = sentences[train_size:]

        print("Train size:", len(train_data))
        print("Test size :", len(test_data))

        train_data, vocab = preproc_sentences(train_data, min_vocab_freq=10)
        test_data, _ = preproc_sentences(test_data, vocab=vocab)

        model = build_n_gram(train_data, vocab, n=3)

        del train_data

        test_data_concat = []

        for v in test_data:
            test_data_concat.extend(v)

        print("Perplexity:", 2 ** log_perplexity(test_data_concat, model))

        with open(filename, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Using {n}-gram model.")
    seq = "this group is"

    seq = preproc_sentences([seq], vocab=set(model["sorted_vocab"]))[0][0]

    print("Initial seq:", seq)

    while seq[-1] != "</s>" and len(seq) <= 128:
        seq.append(autocomplete(seq, model=model, random=False)[0])

    print("Autocompleted seq:", " ".join(seq[:-1]))


if __name__ == "__main__":
    _test()

import typing as t
import re
import random
import collections

import nltk

WORD_REG = re.compile(r"\w+")
letters = "abcdefghijklmnopqrstuvwxyz"

assert len(letters) == 26


def get_vocabulary(corpus, prob: bool = True):

    if isinstance(corpus, str):
        vocab = collections.Counter(WORD_REG.findall(corpus.lower()))

    else:
        vocab = collections.Counter()

        for item in map(str.lower, corpus):
            vocab += collections.Counter(WORD_REG.findall(item))

    if prob:
        total = sum(vocab.values())

        for w, freq in vocab.items():
            vocab[w] = freq / total

    return vocab


def _all_edit_insert(word: str) -> t.Set[str]:
    all_edits = {word[:i] + c + word[i:] for i in range(len(word) + 1) for c in letters}
    return all_edits


def _all_edit_delete(word: str) -> t.Set[str]:
    all_edits = {word[:i] + word[i + 1 :] for i in range(len(word))}
    return all_edits


def _all_edit_replace(word: str) -> t.Set[str]:
    all_edits = {
        word[:i] + c + word[i + 1 :] for i in range(len(word)) for c in letters
    }

    all_edits.discard(word)

    return all_edits


def get_n_edits(word: str, n: int = 1) -> t.Set[str]:
    if n <= 0:
        return {word}

    all_edits = set.union(
        _all_edit_insert(word),
        _all_edit_delete(word),
        _all_edit_replace(word),
    )

    if n > 1:
        all_edits = set.union(*map(lambda item: get_n_edits(item, n - 1), all_edits))

    return all_edits


def get_corrections(word: str, probs: t.Dict[str, float], n: int = 3) -> t.List[str]:
    assert n > 0

    if word in probs:
        return [word]

    candidates = get_n_edits(word, n=1).intersection(probs) or get_n_edits(
        word, n=2
    ).insersection(probs)

    candidates = sorted(
        [(w, probs[w]) for w in candidates], key=lambda item: item[1], reverse=True
    )

    return candidates[:n]


def get_corpus(which: str = "twitter"):
    if which == "twitter":
        tweets_pos = nltk.corpus.twitter_samples.strings("positive_tweets.json")
        tweets_neg = nltk.corpus.twitter_samples.strings("negative_tweets.json")
        return tweets_pos + tweets_neg

    if which == "shakespeare":
        with open("corpus/shakespeare.txt") as f:
            corpus = f.read()

        return corpus

    raise ValueError(f"Unknown corpus '{which}'.")


def _test():
    random.seed(16)

    corpus = get_corpus("twitter")
    probs = get_vocabulary(corpus)

    print("Vocabularity size:", len(probs))
    print("Vocabulary sample:", probs.most_common(5))

    word = "dys"

    print("Insert:", word)
    print(get_corrections(word, probs))


if __name__ == "__main__":
    _test()

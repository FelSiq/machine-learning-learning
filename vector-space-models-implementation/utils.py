import typing as t
import collections

import regex


class BaseFreqs:
    def __init__(self, lowercase: bool = False):
        self.freqs = collections.Counter()
        self.lowercase = bool(lowercase)

    def __setitem__(self, x, v):
        self.freqs[x] = v

    def __len__(self):
        return len(self.freqs)

    def __iter__(self):
        return iter(self.freqs.keys())

    def __getitem__(self, i):
        return self.freqs[i]

    def __contains__(self, x: str):
        if self.lowercase:
            x = x.lower()

        return x in self.freqs

    def __str__(self):
        return str(self.freqs)

    def _preprocess_tokens(self, X):
        if isinstance(X, str):
            tokens = regex.findall(r"(?u)\b\w\w+\b", X)

        else:
            tokens = X

        tokens = [token for token in tokens if token]

        if self.lowercase:
            tokens = list(map(str.lower, tokens))

        return tokens


def get_text(filepath: str) -> str:
    if isinstance(filepath, int):
        text = {
            0: "Hello, this is an arbitrary text! :)\n This is a (not so useful) newline in this text?",
            1: "I don't know if this is an good text for template, but maybe this text is original.",
        }[filepath]
        return text

    with open(filepath, "r") as f:
        text = f.read().strip()

    return text

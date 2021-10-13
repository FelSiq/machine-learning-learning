import typing as t
import collections

import regex


class BaseFreqs:
    def __init__(self, lowercase: bool = False):
        self.lowercase = bool(lowercase)

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

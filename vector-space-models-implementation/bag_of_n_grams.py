import typing as t
import collections

import numpy as np

import utils


class TermFrequency(utils.BaseFreqs):
    def __init__(
        self, n_gram: int = 1, norm: t.Optional[int] = None, lowercase: bool = False
    ):
        assert int(n_gram) >= 1
        assert norm is None or float(norm) > 0.0

        super(TermFrequency, self).__init__(lowercase=lowercase)

        self.n_gram = n_gram
        self.norm = norm if norm is None else float(norm)

        self.freqs = collections.Counter()

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

    def _get_ngram_freqs(
        self, tokens: t.Sequence[str]
    ) -> t.Sequence[t.Union[str, t.Tuple[str]]]:
        freqs = collections.Counter()

        for i in range(max(0, len(tokens) - self.n_gram) + 1):
            ngram = tuple(tokens[i : i + self.n_gram]) if self.n_gram > 1 else tokens[i]
            freqs[ngram] += 1

        return freqs

    def fit(self, X: t.Union[t.Sequence[str], str]):
        tokens = self._preprocess_tokens(X)
        freqs = self._get_ngram_freqs(tokens)

        if self.norm is not None:
            total_freq = np.maximum(
                1e-7, np.linalg.norm(list(freqs.values()), ord=self.norm)
            )
            freqs = {term: freq / total_freq for term, freq in freqs.items()}

        self.freqs = freqs

        return self


def _test():
    import sklearn.feature_extraction

    text = utils.get_text(0)
    model = TermFrequency(norm=2.0, lowercase=True, n_gram=2).fit(text)

    ref_a = sklearn.feature_extraction.text.CountVectorizer(
        ngram_range=(2, 2)
    ).fit_transform([text])
    ref_b = sklearn.feature_extraction.text.TfidfVectorizer(
        norm="l2",
        use_idf=False,
        ngram_range=(2, 2),
    ).fit_transform([text])

    print(model)
    print()
    print(ref_a)
    print()
    print(ref_b)


if __name__ == "__main__":
    _test()

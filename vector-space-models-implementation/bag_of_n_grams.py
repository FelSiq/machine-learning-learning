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
    model = TermFrequency(norm=None, lowercase=True).fit(text)

    ref_a = sklearn.feature_extraction.text.CountVectorizer().fit_transform([text])
    ref_b = sklearn.feature_extraction.text.TfidfVectorizer(
        norm="l2", use_idf=False
    ).fit_transform([text])

    print(model)
    print()
    print(ref_a)
    print()
    print(ref_b)


if __name__ == "__main__":
    _test()

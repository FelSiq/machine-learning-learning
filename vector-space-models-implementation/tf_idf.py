import typing as t
import collections

import numpy as np

import utils
import bag_of_n_grams


class TFIDF(utils.BaseFreqs):
    def __init__(self, n_gram: int = 1, lowercase: bool = False, smooth: bool = True, norm: t.Optional[float] = 2.0):
        assert int(n_gram) >= 1
        assert norm is None or float(norm) > 0.0

        super(TFIDF, self).__init__(lowercase=lowercase)

        self.n_gram = int(n_gram)
        self.tf_idfs = []  # type: t.List[utils.TermFrequency]
        self.vocabulary = frozenset()
        self.smooth = bool(smooth)
        self.norm = norm if norm is None else float(norm)

    def __str__(self):
        return "\n".join([str(tf_idf) for tf_idf in self.tf_idfs])

    def fit(self, X: t.Sequence[t.Union[str, t.Sequence[str]]]):
        doc_tokens = [self._preprocess_tokens(x) for x in X]

        self.tf_idfs = []  # type: t.List[utils.TermFrequency]
        doc_freqs = collections.Counter()

        for tokens in doc_tokens:
            tf = bag_of_n_grams.TermFrequency(
                n_gram=self.n_gram,
                lowercase=self.lowercase,
                norm=self.norm,
            ).fit(tokens)

            self.tf_idfs.append(tf)
            doc_freqs.update(tf)

        self.vocabulary = frozenset(doc_freqs)
        n_docs = len(X)
        m = int(self.smooth)

        log_inv_doc_freqs = {
            term: float(1.0 - float(np.log((f + m) / (n_docs + m))))
            for term, f in doc_freqs.items()
        }

        for tf_idf in self.tf_idfs:
            for term in tf_idf:
                tf_idf[term] *= log_inv_doc_freqs[term]

            norm = 1e-7 + float(np.linalg.norm(list(tf_idf.freqs.values()), ord=self.norm))
            tf_idf.freqs = {term: f / norm for term, f in tf_idf.freqs.items()}

        return self


def _test():
    import sklearn.feature_extraction

    texts = [utils.get_text(0), utils.get_text(1)]

    ref = sklearn.feature_extraction.text.TfidfVectorizer().fit_transform(texts)

    model = TFIDF(lowercase=True).fit(texts)

    print(model)
    print(ref)


if __name__ == "__main__":
    _test()

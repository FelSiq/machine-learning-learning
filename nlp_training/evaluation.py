import collections
import typing as t

import numpy as np


def _get_n_grams(seq: t.Sequence[t.Any], n: int):
    n_grams = collections.Counter()

    for i in range(len(seq) - n + 1):
        n_grams[tuple(seq[i : i + n])] += 1

    return n_grams


def _get_intersection(
    counter_a: t.Counter[t.Any], counter_b: t.Counter[t.Any]
) -> t.Counter[t.Any]:
    return collections.Counter(
        {k: min(counter_a[k], counter_b[k]) for k in counter_b if k in counter_a}
    )


def _calc_brevity_penalty(new: t.Sequence[t.Any], ref: t.Sequence[t.Any]):
    return min(1.0, np.exp(1 - len(ref) / len(new)))


def bleu(
    new: t.Sequence[t.Any],
    ref: t.Sequence[t.Any],
    ns: int = 1,
    weights: t.Optional[t.Sequence[float]] = None,
) -> float:
    """Bleu score (Bilingual Evaluation Understudy)."""
    assert len(new) >= ns >= 1
    assert ref

    if weights is None:
        weights = ns * [1.0 / ns]

    assert np.isclose(1.0, sum(weights))

    clipped_precisions = np.zeros(ns, dtype=float)

    for n in np.arange(1, 1 + ns):
        n_gram_new = _get_n_grams(new, n=n)
        n_gram_ref = _get_n_grams(ref, n=n)
        n_gram_intersec = _get_intersection(n_gram_new, n_gram_ref)

        precision = sum(n_gram_intersec.values()) / (len(new) - n + 1)

        clipped_precisions[n - 1] = precision

    avg_bleu_score = np.exp(np.dot(weights, np.log(1e-8 + clipped_precisions)))

    brevity_penalty = _calc_brevity_penalty(new, ref)

    return brevity_penalty * avg_bleu_score


def rouge(
    new: t.Sequence[t.Any], ref: t.Sequence[t.Any], n: int = 1, beta: float = 1.0
) -> float:
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score."""
    count_new = _get_n_grams(new, n=n)
    count_ref = _get_n_grams(ref, n=n)

    sum_new = sum(count_new.values())
    sum_ref = sum(count_ref.values())
    sum_intersection = sum(_get_intersection(count_new, count_ref).values())

    precision = sum_intersection / sum_new
    recall = sum_intersection / sum_ref

    if beta * precision + recall > 0.0:
        fb_score = (1.0 + beta) * precision * recall / (beta * precision + recall)

    else:
        fb_score = 0.0

    return fb_score


def _test():
    import nltk

    reference = (
        "The NASA Opportunity rover is battling a massive dust storm on planet Mars."
    )
    candidate_1 = "The Opportunity rover is combating a big sandstorm on planet Mars."
    candidate_2 = "A NASA rover is fighting a massive storm on planet Mars."

    tokenized_ref = nltk.word_tokenize(reference.lower())
    tokenized_cand_1 = nltk.word_tokenize(candidate_1.lower())
    tokenized_cand_2 = nltk.word_tokenize(candidate_2.lower())

    n = 4

    bleu_score_1 = bleu(tokenized_cand_1, tokenized_ref, ns=n)
    bleu_score_2 = bleu(tokenized_cand_2, tokenized_ref, ns=n)

    print(bleu_score_1)
    print(bleu_score_2)

    rouge_score_1 = rouge(tokenized_cand_2, tokenized_ref, n=n)
    rouge_score_2 = rouge(tokenized_cand_1, tokenized_ref, n=n)

    print(rouge_score_1)
    print(rouge_score_2)


if __name__ == "__main__":
    _test()

import typing as t

import nltk
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

import tweets_utils


def logprobs(freqs, all_words: int):
    total_words = sum(freqs.values())
    num_words = len(all_words)
    # Note: applying laplace smoothing
    probs = {
        w: np.log((freqs.get(w, 0) + 1) / (total_words + num_words)) for w in all_words
    }
    return probs


def calc_lambdas(
    logprobs_pos: t.Dict[str, float], logprobs_neg: t.Dict[str, float]
) -> t.Dict[str, float]:
    lambdas = dict()

    assert len(logprobs_pos) == len(logprobs_neg)

    for w in logprobs_pos:
        lambdas[w] = logprobs_pos[w] - logprobs_neg[w]

    return lambdas


def _test(train_size: int = 4500):
    (
        train_tweets,
        train_labels,
        test_tweets,
        test_labels,
        freq_pos,
        freq_neg,
    ) = tweets_utils.get_data(train_size)

    all_words = freq_pos.keys() | freq_neg

    logprobs_pos = logprobs(freq_pos, all_words)
    logprobs_neg = logprobs(freq_neg, all_words)

    lambdas = calc_lambdas(logprobs_pos, logprobs_neg)

    pos_num = np.mean(train_labels == 1)

    log_prior = np.log(pos_num / (1 - pos_num))

    preds = np.zeros(test_labels.size)

    for i, tweet in enumerate(test_tweets):
        pred = log_prior

        for w in tweet:
            pred += lambdas.get(w, 0)

        preds[i] = int(pred >= 0.0)

    acc = sklearn.metrics.accuracy_score(preds, test_labels.ravel())
    print(f"Test accuracy: {acc:.6f}")

    train_probs_plot_pos = np.zeros(train_labels.size)
    train_probs_plot_neg = np.zeros(train_labels.size)

    for i, tweet in enumerate(train_tweets):
        pred_pos = pred_neg = log_prior

        for w in tweet:
            pred_pos += logprobs_pos.get(w, 0)
            pred_neg += logprobs_neg.get(w, 0)

        train_probs_plot_pos[i] = pred_pos
        train_probs_plot_neg[i] = pred_neg

    colors = ["red", "purple"]
    plt.figure(figsize=(10, 10))
    plt.scatter(
        train_probs_plot_pos,
        train_probs_plot_neg,
        c=[colors[cls] for cls in map(int, train_labels)],
        s=0.1,
    )
    plt.tight_layout()
    plt.axis("off")
    plt.savefig("cute_graph_2")
    plt.show()


if __name__ == "__main__":
    _test()

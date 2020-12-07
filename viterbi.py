import typing as t
import collections
import itertools
import pickle

import numpy as np

import viterbi_utils_pos


def hidden_markov_model(
    tagged_corpus: t.List[t.Tuple[str, str]],
    vocab: t.Dict[str, int],
    initial_tag: str = "--s--",
    smoothing_factor: float = 1e-3,
) -> t.Dict[str, t.Any]:
    assert smoothing_factor >= 0

    count_tag = collections.Counter()
    count_tag_tag = collections.Counter()
    count_tag_word = collections.Counter()

    prev_tag = initial_tag

    for word_tag in tagged_corpus:
        word, tag = viterbi_utils_pos.get_word_tag(word_tag, vocab)
        key_tag_tag = (prev_tag, tag)
        key_tag_word = (tag, word)

        count_tag[tag] += 1
        count_tag_tag[key_tag_tag] += 1
        count_tag_word[key_tag_word] += 1

        prev_tag = tag

    sorted_tags = sorted(count_tag)
    num_tags = len(sorted_tags)
    num_words = len(vocab)

    mat_transition = np.zeros((num_tags, num_tags), dtype=float)
    mat_emission = np.zeros((num_tags, num_words), dtype=float)

    inds_tag = np.arange(num_tags)
    inds_word = np.arange(num_words)

    for i, j in itertools.product(inds_tag, inds_tag):
        n_transit = count_tag_tag[(sorted_tags[i], sorted_tags[j])]
        n_tag_prev = count_tag[sorted_tags[i]]
        mat_transition[i, j] = (n_transit + smoothing_factor) / (
            n_tag_prev + num_tags * smoothing_factor
        )

    for i, w in itertools.product(inds_tag, vocab):
        n_emission = count_tag_word[(sorted_tags[i], w)]
        n_tag = count_tag[sorted_tags[i]]
        j = vocab[w]
        mat_emission[i, j] = (n_emission + smoothing_factor) / (
            n_tag + num_words * smoothing_factor
        )

    model = {
        "mat_transition": mat_transition,
        "mat_emission": mat_emission,
        "sorted_tags": sorted_tags,
        "initial_tag": initial_tag,
        "initial_tag_ind": sorted_tags.index(initial_tag),
    }

    return model


def viterbi(words: t.List[str], model: t.Dict[str, t.Any], vocab: t.Dict[str, int]):
    if isinstance(words, str):
        words = words.split()

    mat_transition = model["mat_transition"]
    mat_emission = model["mat_emission"]
    sorted_tags = model["sorted_tags"]
    initial_tag_ind = model["initial_tag_ind"]

    num_tags = len(sorted_tags)
    num_words = len(words)

    tags = num_words * [None]

    probs = np.zeros((2, num_tags))

    for i in np.arange(num_tags):
        probs[0, i] = (
            (
                np.log(mat_transition[initial_tag_ind, i])
                + np.log(mat_emission[i, vocab[words[0]]])
            )
            if np.not_equal(mat_transition[initial_tag_ind, i], 0)
            else -np.inf
        )

    for i in np.arange(1, num_words):
        for j in np.arange(num_tags):
            best_prob_i = -np.inf
            best_tag_i = None

            for k in np.arange(num_tags):
                prob = (
                    probs[(i - 1) % 2, k]
                    + np.log(mat_transition[k, j])
                    + np.log(mat_emission[j, vocab[words[i]]])
                )
                if best_prob_i < prob:
                    best_prob_i = prob
                    best_tag_i = k

            probs[i % 2, j] = best_prob_i
            tags[i - 1] = sorted_tags[best_tag_i]

    tags[-1] = sorted_tags[np.argmax(probs[(1 + num_words) % 2, :])]

    return tags


def get_vocab(path: str = "corpus/hmm_vocab.txt") -> t.Dict[str, int]:
    vocab = dict()

    with open(path) as f:
        words = f.read().split("\n")

    for i, w in enumerate(sorted(words)):
        vocab[w] = i

    return vocab


def _test():
    pickle_file = "hmm_viterbi_model.pickle"

    vocab = get_vocab()
    print(f"Got vocab (len = {len(vocab)}).")

    try:
        with open(pickle_file) as f:
            model = pickle.load(f)

        print("Loaded pre-saved HMM.")

    except FileNotFoundError:
        with open("corpus/wsj_train.pos") as f:
            tagged_corpus_train = f.readlines()
            print("Got train corpus for HMM.")

        with open("corpus/wsj_test.pos") as f:
            tagged_corpus_test = f.readlines()

        _, test_words = viterbi_utils_pos.preprocess(vocab, "corpus/test_words.txt")
        print("Got test corpus for HMM.")

        model = hidden_markov_model(tagged_corpus_train, vocab)
        print("Built HMM.")

        tags_pred = viterbi(test_words, model, vocab)
        print("Finished Viterbi.")

        correct = total = 0

        for tag_pred, word_tag in zip(tags_pred, tagged_corpus_test):
            word_tag = word_tag.split()

            if len(word_tag) != 2:
                continue

            word, tag_true = word_tag

            if tag_pred == tag_true:
                correct += 1

            total += 1

        test_accuracy = correct / total

        print(f"Test accuracy: {test_accuracy:.4f}")

        assert test_accuracy >= 0.95

        with open(pickle_file, "w") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved HMM in pickle file.")

    with open("corpus/test_words.txt") as f:
        test_words = f.read().split("\n")

    tags = viterbi(test_words, model, vocab)

    print(test_words[:6])
    print(tags[:6])


if __name__ == "__main__":
    _test()

import collections
import itertools

import numpy
import trax
import trax.fastmath.numpy as np
import nltk
import sklearn.model_selection

import tweets_utils


def build_vocab(
    X,
    min_freq: int = 3,
    unk_token: str = "__UNK__",
    pad_token: str = "__PAD__",
    end_token: str = "__</e>__",
):
    freqs = collections.Counter()

    for tokens in X:
        freqs.update(tokens)

    vocab = {pad_token, end_token, unk_token}
    vocab.update({k for k, v in freqs.items() if v >= min_freq})

    return dict(zip(vocab, range(len(vocab))))


def get_pad_size(X):
    return 1 + max(map(len, X))


def pad_insts(X, size: int, pad_token: str = "__PAD__", end_token: str = "__</e>__"):
    for i in range(len(X)):
        X[i].append(end_token)

        rem = size - len(X[i])

        if rem < 0:
            X[i][:-rem]

        elif rem > 0:
            pad = rem * [pad_token]
            X[i].extend(pad)


def str_to_int(X, vocab, unk_token: str = "__UNK__"):
    for i, x in enumerate(X):
        for j, token in enumerate(x):
            x[j] = vocab.get(token, vocab[unk_token])


def get_model(vocab_size: int, embedding_dim: int = 256, output_dim: int = 2):
    model = trax.layers.Serial(
        trax.layers.Embedding(vocab_size=vocab_size, d_feature=embedding_dim),
        trax.layers.Mean(axis=1),
        trax.layers.Dense(n_units=output_dim),
        trax.layers.LogSoftmax(),
    )

    return model


def _test():
    X_train, y_train, X_eval, y_eval, _, _ = tweets_utils.get_data(train_size=4096)

    vocab = build_vocab(X_train)

    pad_size = get_pad_size(X_train)

    pad_insts(X_train, pad_size)
    pad_insts(X_eval, pad_size)

    str_to_int(X_train, vocab)
    str_to_int(X_eval, vocab)

    X_train = np.array(X_train)
    X_eval = np.array(X_eval)

    y_train = np.array(y_train).ravel()
    y_eval = np.array(y_eval).ravel()

    assert X_train.ndim == X_eval.ndim == 2
    assert X_train.shape[1] == X_eval.shape[1]

    model = get_model(len(vocab))

    def batch(X, y, batch_size: int = 16):
        n_splits = y.size // batch_size

        splitter = sklearn.model_selection.StratifiedKFold(
            n_splits=n_splits, shuffle=True
        )

        for _, inds in splitter.split(X, y):
            yield X[inds, :], y[inds], np.ones_like(inds)

    train_task = trax.supervised.training.TrainTask(
        labeled_data=itertools.cycle(batch(X_train, y_train)),
        loss_layer=trax.layers.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=10,
    )

    eval_task = trax.supervised.training.EvalTask(
        labeled_data=itertools.cycle(batch(X_eval, y_eval)),
        metrics=(trax.layers.CrossEntropyLoss(), trax.layers.Accuracy()),
    )

    train_loop = trax.supervised.training.Loop(
        model,
        train_task,
        eval_tasks=eval_task,
        output_dir="dir_trax_sentiment_analysis",
    )
    train_loop.run(n_steps=128)

    preds = train_loop.eval_model(X_eval).argmax(axis=1)

    print("Eval accuracy:", np.mean(preds == y_eval))


if __name__ == "__main__":
    _test()

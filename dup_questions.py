import functools
import random
import collections

import nltk
import trax
import trax.fastmath.numpy as np
import pandas as pd
import numpy


stemmer = nltk.stem.PorterStemmer()


def get_tokens(question: str, stemming: bool = True):
    tokens = nltk.word_tokenize(question)

    if stemming:
        tokens = [stemmer.stem(w) for w in tokens]

    return tokens


def get_corpus(test_size: int = 512):
    assert test_size % 2 == 0

    data = pd.read_csv("corpus/questions.csv", index_col=0)
    dup_inds = data["is_duplicate"] != 0

    half_test_size = test_size // 2

    test_non_duplicate = data.loc[~dup_inds, ["question1", "question2"]].values[
        :half_test_size
    ]

    data = data.loc[dup_inds, ["question1", "question2"]].values

    Q1 = []
    Q2 = []

    for row in data[:-half_test_size]:
        tokens1 = get_tokens(row[0])
        tokens2 = get_tokens(row[1])

        Q1.append(tokens1)
        Q2.append(tokens2)

    assert len(Q1) == len(Q2) == len(data) - half_test_size

    Q1_test = []
    Q2_test = []
    y_test = []

    for row in data[-half_test_size:]:
        tokens1 = get_tokens(row[0])
        tokens2 = get_tokens(row[1])

        Q1_test.append(tokens1)
        Q2_test.append(tokens2)

        y_test.append(1)

    for row in test_non_duplicate:
        tokens1 = get_tokens(row[0])
        tokens2 = get_tokens(row[1])

        Q1_test.append(tokens1)
        Q2_test.append(tokens2)

        y_test.append(0)

    del data

    assert len(Q1_test) == len(Q2_test) == test_size

    return Q1, Q2, Q1_test, Q2_test, np.array(y_test, dtype=np.int32)


def data_generator(Q1, Q2, pad_num: int, batch_size: int = 128, shuffle: bool = True):
    assert len(Q1) == len(Q2)
    assert len(Q1) >= batch_size > 0

    inds = numpy.arange(len(Q1))
    idx = 0

    if shuffle:
        numpy.random.shuffle(inds)

    q1, q2 = [], []
    max_len = 0

    while True:
        if idx >= inds.size:
            idx = 0
            if shuffle:
                numpy.random.shuffle(inds)

        q1.append(Q1[inds[idx]])
        q2.append(Q2[inds[idx]])

        max_len = max(max_len, len(q1[-1]), len(q2[-1]))
        idx += 1

        if len(q1) == batch_size:
            max_len = 2 ** int(np.ceil(np.log2(max_len)))

            for cur_q1, cur_q2 in zip(q1, q2):
                cur_q1.extend((max_len - len(cur_q1)) * [pad_num])
                cur_q2.extend((max_len - len(cur_q2)) * [pad_num])

            yield np.array(q1), np.array(q2)

            q1, q2 = [], []
            max_len = 0


def triplet_loss(v1, v2, margin: float = 0.25):
    cossine = np.dot(v1, v2.T)

    batch_size = cossine.shape[0]

    positive = np.diagonal(cossine)

    neg_mean = np.sum(cossine * (1.0 - np.eye(batch_size)), axis=1) / (batch_size - 1)
    nearest_neg = np.max(cossine - 2.0 * np.eye(batch_size), axis=1)

    loss1 = np.maximum(0.0, neg_mean - positive + margin)
    loss2 = np.maximum(0.0, nearest_neg - positive + margin)

    loss = np.mean(loss1 + loss2)

    return loss


def build_siamese_model(vocab_dim: int, embed_dim: int = 128):
    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    model_seq = trax.layers.Serial(
        trax.layers.Embedding(vocab_dim, embed_dim),
        trax.layers.LSTM(embed_dim),
        trax.layers.Mean(axis=1),
        trax.layers.Fn("Normalize", normalize),
    )

    model_siamese = trax.layers.Parallel(model_seq, model_seq)

    return model_siamese


def build_vocab(
    Q1,
    Q2,
    pad_token: str = "__PAD__",
    unknown_token: str = "__UNK__",
    min_freq: int = 1,
):
    assert min_freq >= 1

    freqs = collections.Counter()

    for q1, q2 in zip(Q1, Q2):
        freqs.update(q1)
        freqs.update(q2)

    vocab = [k for k, f in freqs.items() if f >= min_freq]
    vocab.extend([pad_token, unknown_token])
    vocab.sort()

    return dict(zip(vocab, range(len(vocab))))


def encode_tokens(Q, vocab, unknown_token: str = "__UNK__"):
    for q in Q:
        for i, token in enumerate(q):
            q[i] = vocab.get(token, vocab[unknown_token])


def predict(
    model,
    Q1_test,
    Q2_test,
    vocab,
    pad_token: str = "__PAD__",
    batch_size: int = 128,
    threshold: float = 0.7,
):
    test_generator = data_generator(
        Q1_test, Q2_test, pad_num=vocab[pad_token], batch_size=batch_size, shuffle=False
    )

    test_size = len(Q1_test)

    preds = []

    for i in np.arange(0, test_size, batch_size):
        q1_batch, q2_batch = next(test_generator)
        v1, v2 = model([q1_batch, q2_batch])
        for j in np.arange(batch_size):
            cossine = np.dot(v1[j], v2[j].T)
            preds.append(cossine >= threshold)

    return np.array(preds, dtype=np.int32)


def _test():
    pad_token = "__PAD__"
    margin = 0.25
    output_dir = "dir_dup_questions_stemming_very_freq_only"
    n_steps = 256
    test_size = 512
    train = True

    Q1, Q2, Q1_test, Q2_test, y_test = get_corpus(test_size)

    print("Question pairs num:", len(Q1))

    train_size = int(0.95 * len(Q1))

    print("Train size:", train_size)
    print("Eval size :", len(Q1) - train_size)
    print("Test size :", test_size)

    Q1_train, Q2_train = Q1[:train_size], Q2[:train_size]
    Q1_eval, Q2_eval = Q1[train_size:], Q2[train_size:]

    assert len(Q1_test) == len(Q2_test) == test_size
    assert len(Q1_eval) == len(Q2_eval) == len(Q1) - train_size
    assert len(Q1_train) == len(Q2_train) == train_size

    del Q1
    del Q2

    print("Tokens (train) sample:", Q1_train[0])
    print("Tokens (eval) sample :", Q1_eval[0])
    print("Tokens (test) sample :", Q1_test[0])

    vocab = build_vocab(Q1_train, Q2_train, pad_token=pad_token, min_freq=3)

    print("Vocab len:", len(vocab))

    encode_tokens(Q1_train, vocab)
    encode_tokens(Q2_train, vocab)
    encode_tokens(Q1_eval, vocab)
    encode_tokens(Q2_eval, vocab)
    encode_tokens(Q1_test, vocab)
    encode_tokens(Q2_test, vocab)

    print("Encoded tokens (train) sample:", Q1_train[0])
    print("Encoded tokens (eval) sample :", Q1_eval[0])
    print("Encoded tokens (test) sample :", Q1_test[0])

    model = build_siamese_model(len(vocab))

    print(model)

    if train:
        train_generator = data_generator(
            Q1_train, Q2_train, pad_num=vocab[pad_token], batch_size=256
        )
        eval_generator = data_generator(
            Q1_eval, Q2_eval, pad_num=vocab[pad_token], batch_size=256
        )

        def TripletLoss(margin: float = 0.25):
            l = trax.layers.Fn(
                "TripletLoss", functools.partial(triplet_loss, margin=margin)
            )

            return l

        lr_schedule = trax.lr.warmup_and_rsqrt_decay(400, 0.01)

        train_task = trax.supervised.training.TrainTask(
            labeled_data=train_generator,
            loss_layer=TripletLoss(),
            optimizer=trax.optimizers.Adam(0.01),
            n_steps_per_checkpoint=10,
            lr_schedule=lr_schedule,
        )

        eval_task = trax.supervised.training.EvalTask(
            labeled_data=eval_generator,
            metrics=[TripletLoss()],
        )

        train_loop = trax.supervised.training.Loop(
            model,
            train_task,
            eval_tasks=eval_task,
            output_dir=output_dir,
        )

        train_loop.run(n_steps=n_steps)

    else:
        import os

        model.init_from_file(os.path.join(output_dir, "model.pkl.gz"))
        print("Loaded model.")

    preds = predict(model, Q1_test, Q2_test, vocab)

    print(preds)

    acc_test = np.mean(preds == y_test[: preds.size])

    print(f"Test accuracy: {acc_test:.4f}")


if __name__ == "__main__":
    _test()

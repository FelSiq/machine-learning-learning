import functools

import nltk
import trax
import trax.fastmath.numpy as np
import pandas as pd
import numpy


def get_corpus(n=None):
    data = pd.read_csv("corpus/questions.csv", index_col=0, nrows=n)
    dup_inds = data["is_duplicate"] != 0
    data = data.loc[dup_inds, ["question1", "question2"]].values

    Q1 = []
    Q2 = []

    for row in data:
        tokens1 = nltk.word_tokenize(row[0])
        tokens2 = nltk.word_tokenize(row[1])

        Q1.append(tokens1)
        Q2.append(tokens2)

    return Q1, Q2


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

        if idx == batch_size:
            max_len = 2 ** int(np.ceil(np.log2(max_len)))

            q1_batch = []
            q2_batch = []

            for j in np.arange(batch_size):
                q1_batch.append(q1[j] + (max_len - len(q1[j])) * [pad_num])
                q2_batch.append(q2[j] + (max_len - len(q2[j])) * [pad_num])

            yield np.array(q1_batch), np.array(q2_batch)

            q1, q2 = [], []
            max_len = 0


def triplet_loss(v1, v2, margin: 0.25):
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


def build_vocab(Q1, Q2, pad_token: str = "__PAD__", unknown_token: str = "__UNK__"):
    vocab = {pad_token, unknown_token}

    for q1, q2 in zip(Q1, Q2):
        vocab.update(q1)
        vocab.update(q2)

    return dict(zip(vocab, range(len(vocab))))


def encode_tokens(Q, vocab, unknown_token: str = "__UNK__"):
    for q in Q:
        for i, token in enumerate(q):
            q[i] = vocab.get(token, vocab[unknown_token])


def _test():
    pad_token = "__PAD__"
    margin = 0.25
    output_dir = "dir_dup_questions"
    n_steps = 256
    test_size = 512

    Q1, Q2 = get_corpus()

    print("Question pairs num:", len(Q1))

    train_size = int(0.95 * len(Q1))

    print("Train size:", train_size)
    print("Eval size :", len(Q1) - train_size - test_size)
    print("Test size :", test_size)

    Q1_train, Q2_train = Q1[:train_size], Q2[:train_size]
    Q1_eval, Q2_eval = Q1[train_size:test_size], Q2[train_size:test_size]
    Q1_test, Q2_test = Q1[-test_size:], Q2[-test_size:]

    del Q1
    del Q2

    assert len(Q1_test) == len(Q2_test) == test_size
    assert len(Q1_eval) == len(Q2_eval) == len(Q1) - train_size - test_size
    assert len(Q1_train) == len(Q2_train) == train_size

    vocab = build_vocab(Q1_train, Q2_train, pad_token=pad_token)

    print("Vocab len:", len(vocab))

    encode_tokens(Q1_train, vocab)
    encode_tokens(Q2_train, vocab)
    encode_tokens(Q1_eval, vocab)
    encode_tokens(Q2_eval, vocab)

    print("Encoded tokens (train) sample:", Q1_train[0])
    print("Encoded tokens (eval) sample:", Q1_eval[0])

    model = build_siamese_model(len(vocab))

    print(model)

    train_generator = data_generator(
        Q1_train, Q2_train, pad_num=vocab[pad_token], batch_size=256
    )
    eval_generator = data_generator(
        Q1_eval, Q2_eval, pad_num=vocab[pad_token], batch_size=256
    )

    TripletLoss = trax.layers.Fn(
        "TripletLoss", functools.partial(triplet_loss, margin=margin)
    )

    lr_schedule = trax.lr.warmup_and_rsqrt_decay(400, 0.01)

    train_task = trax.supervised.training.TrainTask(
        labeled_data=train_generator,
        loss_layer=TripletLoss,
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=10,
        lr_schedule=lr_schedule,
    )

    eval_task = trax.supervised.training.EvalTask(
        labeled_data=eval_generator,
        metrics=[TripletLoss],
    )

    train_loop = trax.supervised.training.Loop(
        model,
        train_task,
        eval_tasks=eval_task,
        output_dir=output_dir,
    )

    train_loop.run(n_steps=n_steps)


if __name__ == "__main__":
    _test()

import typing as t
import re
import bisect
import collections
import pickle

import numpy as np
import nltk
import emoji
import torch
import torch.nn as nn


full_stop_reg = re.compile(r"[;,!?-]+")

emoji_reg = emoji.get_emoji_regexp()

tokenizer = nltk.tokenize.TweetTokenizer(
    preserve_case=False, strip_handles=True, reduce_len=True
)


class MLP(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(MLP, self).__init__()
        self.weights = nn.Sequential(
            nn.Linear(vocab_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, vocab_size),
        )

    def forward(self, X):
        return self.weights(X)


def sliding_window(tokens: t.Sequence[str], C: int = 2):
    assert C > 0

    i = C

    while i < len(tokens) - C:
        word = tokens[i]
        context = tokens[i - C : i] + tokens[i + 1 : i + 1 + C]
        yield context, word
        i += 1


def preprocess_sentence(sentence: str):
    sentence = full_stop_reg.sub(".", sentence)

    tokens = tokenizer.tokenize(sentence)

    tokens = [
        token
        for token in tokens
        if token.isalpha() or token == "." or emoji_reg.search(token)
    ]

    return tokens


def get_X_y(tokens: t.Sequence[str], sorted_vocab: t.Sequence[str], C: int = 2):
    for context, word in sliding_window(tokens, C):
        tg_ind = bisect.bisect_left(sorted_vocab, word)

        if tg_ind >= len(sorted_vocab):
            continue

        X = np.zeros(len(sorted_vocab), dtype=np.float32)
        y = np.zeros(len(sorted_vocab), dtype=np.uint32)

        freqs = collections.Counter(context)

        for w, freq in freqs.items():
            ind = bisect.bisect_left(sorted_vocab, w)
            if ind < len(X):
                X[ind] += freq

        X /= np.sum(X)

        y[tg_ind] = 1

        yield X, y


def get_tokens(C: int = 2, freq_min: int = 5):
    assert freq_min >= 1

    tweets_pos = nltk.corpus.twitter_samples.strings("positive_tweets.json")
    tweets_neg = nltk.corpus.twitter_samples.strings("negative_tweets.json")

    tweets = tweets_pos + tweets_neg

    tweets = tweets[:20]

    word_freqs = collections.Counter()

    for i, tweet in enumerate(tweets):
        tweets[i] = tokens = preprocess_sentence(tweet)
        word_freqs.update(tokens)

    sorted_vocab = sorted({k for k, v in word_freqs.items() if v >= freq_min})

    X = []
    y = []

    for tokens in tweets:
        for x_i, y_i in get_X_y(tokens, sorted_vocab, C=C):
            X.append(x_i)
            y.append(y_i)

    return X, y, sorted_vocab


def _test():
    nltk.download("punkt")
    nltk.download("twitter_samples")
    embedding_dim = 512
    C = 2
    device = "cuda"

    X, y, sorted_vocab = get_tokens(C=C)

    X = torch.tensor(X, dtype=torch.float32, device=device)
    y_inds = torch.tensor(y, dtype=torch.long, device=device).argmax(axis=1)

    model = MLP(vocab_size=len(sorted_vocab), embedding_dim=embedding_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    for i in np.arange(400):
        optim.zero_grad()
        preds = model(X)
        loss = criterion(preds, y_inds)
        loss.backward()
        optim.step()

        if (1 + i) % 50 == 0:
            print(f"{1 + i}: {loss.item():.4f}")

    params = dict(model.named_parameters())

    W0 = params["weights.0.weight"]
    W1 = params["weights.2.weight"]

    embeddings = 0.5 * (W0.T + W1).detach().cpu().numpy()

    assert embeddings.shape == (len(sorted_vocab), embedding_dim)

    model = {
        "embeddings": embeddings,
        "sorted_vocab": sorted_vocab,
    }

    with open(
        f"embedding_models/embedding_model_{C}_{embedding_dim}.pickle", "wb"
    ) as f:
        pickle.dump(
            model,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    _test()

import typing as t
import collections
import functools

import torch
import torch.nn as nn
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import sklearn.metrics


class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Linear(3, 1)

    def forward(self, features):
        return self.weights(features)


lang = "english"


reg_pipeline = [
    re.compile(r"^RT[\s]+"),
    re.compile(r"https?:\/\/.*[\r\n]*"),
    re.compile(r"#"),
]


nltk.download("twitter_samples")
nltk.download("stopwords")

tokenizer = nltk.tokenize.TweetTokenizer(
    preserve_case=False, strip_handles=True, reduce_len=True
)
stemmer = nltk.stem.PorterStemmer()

stop_words_eng = frozenset(nltk.corpus.stopwords.words(lang))
punctuation = frozenset(string.punctuation)


def process_tweet(tweet: str) -> t.List[str]:
    for reg in reg_pipeline:
        tweet = reg.sub("", tweet)

    words = tokenizer.tokenize(tweet)

    words = [
        stemmer.stem(w)
        for w in words
        if w not in stop_words_eng and w not in punctuation
    ]

    return words


def process_all_tweets(tweets) -> t.List[t.List[str]]:
    return list(map(process_tweet, tweets))


def word_count(tweets_proc: t.List[t.List[str]]) -> t.Dict[str, int]:
    return functools.reduce(lambda x, y: x + y, map(collections.Counter, tweets_proc))


def extract_features(
    tweets_proc: t.List[t.List[str]],
    freqs_pos: t.Dict[str, int],
    freqs_neg: t.Dict[str, int],
):
    feats = np.ones((len(tweets_proc), 3))

    for i, tweet_uniques in enumerate(map(set, tweets_proc)):
        feats[i][1] = sum(freqs_pos[w] for w in tweet_uniques)
        feats[i][2] = sum(freqs_neg[w] for w in tweet_uniques)

    return feats


def _test(train_size: int = 4500, device: str = "cuda"):
    tweets_pos = nltk.corpus.twitter_samples.strings("positive_tweets.json")
    tweets_neg = nltk.corpus.twitter_samples.strings("negative_tweets.json")

    tweets_pos_proc = process_all_tweets(tweets_pos)
    tweets_neg_proc = process_all_tweets(tweets_neg)

    labels_pos = np.ones((len(tweets_pos_proc), 1))
    labels_neg = np.zeros((len(tweets_neg_proc), 1))

    train_tweets_pos, test_tweets_pos = (
        tweets_pos_proc[:train_size],
        tweets_pos_proc[train_size:],
    )
    train_labels_pos, test_labels_pos = labels_pos[:train_size], labels_pos[train_size:]
    train_tweets_neg, test_tweets_neg = (
        tweets_neg_proc[:train_size],
        tweets_neg_proc[train_size:],
    )
    train_labels_neg, test_labels_neg = labels_neg[:train_size], labels_neg[train_size:]

    assert len(train_tweets_pos) == train_labels_pos.size
    assert len(train_tweets_neg) == train_labels_neg.size

    print("train pos len :", len(train_tweets_pos))
    print("train pos len :", len(train_tweets_neg))
    print("test neg len  :", len(test_tweets_neg))
    print("test neg len  :", len(test_tweets_neg))

    freq_pos = word_count(train_tweets_pos)
    freq_neg = word_count(train_tweets_neg)

    train_feat_pos = extract_features(train_tweets_pos, freq_pos, freq_neg)
    train_feat_neg = extract_features(train_tweets_neg, freq_pos, freq_neg)
    test_feat_pos = extract_features(test_tweets_pos, freq_pos, freq_neg)
    test_feat_neg = extract_features(test_tweets_neg, freq_pos, freq_neg)

    X_train = np.vstack((train_feat_pos, train_feat_neg))
    X_test = np.vstack((test_feat_pos, test_feat_neg))
    y_train = np.vstack((train_labels_pos, train_labels_neg))
    y_test = np.vstack((test_labels_pos, test_labels_neg))

    inds_train = np.arange(y_train.size)
    np.random.shuffle(inds_train)

    X_train = torch.tensor(X_train[inds_train, :], device=device, dtype=torch.float)
    y_train = torch.tensor(y_train[inds_train], device=device, dtype=torch.float)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    model = LogReg().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=25)
    criterion = nn.BCEWithLogitsLoss()

    losses = np.zeros(100)

    for i in np.arange(100):
        optim.zero_grad()
        y_preds = model(X_train)
        loss = criterion(y_preds, y_train)
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print(f"{i:<{4}} - {loss.item():.6f}")

        losses[i] = loss.item()

    with torch.no_grad():
        y_preds = model(X_test).cpu() >= 0.0
        test_acc = sklearn.metrics.accuracy_score(y_preds, y_test)
        print(f"Test accuracy: {test_acc:.6f}")

    print("Theta:", list(model.parameters()))

    colors = ["red", "green"]
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.subplot(1, 2, 2)
    plt.scatter(
        *X_train.cpu()[:, 1:].T,
        c=[colors[cls] for cls in y_train.cpu().int().squeeze()],
        s=0.1,
    )
    plt.show()


if __name__ == "__main__":
    _test()

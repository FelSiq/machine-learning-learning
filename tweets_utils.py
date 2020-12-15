import typing as t
import collections
import functools
import re
import string

import nltk
import numpy as np


lang = "english"
remove_emoticon = False


reg_pipeline = [
    re.compile(r"^RT[\s]+"),
    re.compile(r"https?:\/\/.*[\r\n]*"),
    re.compile(r"#"),
]

if remove_emoticon:
    reg_pipeline.append(re.compile(r":[^\s]+"))


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


def get_data(train_size: int = 4500):
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

    train_tweets = train_tweets_pos + train_tweets_neg
    train_labels = np.vstack((train_labels_pos, train_labels_neg))

    test_tweets = test_tweets_pos + test_tweets_neg
    test_labels = np.vstack((test_labels_pos, test_labels_neg))

    print(len(train_tweets), train_labels.size)
    print(len(test_tweets), test_labels.size)
    assert len(train_tweets) == train_labels.size
    assert len(test_tweets) == test_labels.size

    return train_tweets, train_labels, test_tweets, test_labels, freq_pos, freq_neg

import typing as t

import torch
import torch.nn as nn
import nltk
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import tweets_utils


device = "cpu"


class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Linear(3, 1)

    def forward(self, features):
        return self.weights(features)


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


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    device: str = "cpu",
) -> t.Tuple[LogReg, np.ndarray]:

    if not torch.is_tensor(X_train):
        X_train = torch.tensor(train_feat, device=device, dtype=torch.float)
        y_train = torch.tensor(train_labels, device=device, dtype=torch.float)

    model = LogReg().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.3, weight_decay=20)
    criterion = nn.BCEWithLogitsLoss()

    losses = np.zeros(300)

    for i in np.arange(300):
        optim.zero_grad()
        y_preds = model(X_train)
        loss = criterion(y_preds, y_train)
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print(f"{i:<{4}} - {loss.item():.6f}")

        losses[i] = loss.item()

    return model, losses


def _test(train_size: int = 4500, device: str = "cuda"):
    (
        train_tweets,
        train_labels,
        test_tweets,
        test_labels,
        freq_pos,
        freq_neg,
    ) = tweets_utils.get_data(train_size)

    train_feat = extract_features(train_tweets, freq_pos, freq_neg)
    test_feat = extract_features(test_tweets, freq_pos, freq_neg)

    X_train = torch.tensor(train_feat, device=device, dtype=torch.float)
    y_train = torch.tensor(train_labels, device=device, dtype=torch.float)
    X_test = torch.tensor(test_feat, device=device, dtype=torch.float)
    y_test = torch.tensor(test_labels, dtype=torch.float)

    model, losses = train_model(X_train, y_train, 300, device=device)

    with torch.no_grad():
        y_preds = model(X_test).cpu() >= 0.0
        test_acc = sklearn.metrics.accuracy_score(y_preds, y_test)
        print(f"Test accuracy: {test_acc:.6f}")

    print("Theta:", list(model.parameters()))

    # plt.subplot(1, 2, 1)
    # plt.plot(losses)
    # plt.subplot(1, 2, 2)
    plt.figure(figsize=(10, 10))
    colors = ["red", "purple"]
    plt.scatter(
        *X_train.cpu()[:, 1:].T,
        c=[colors[cls] for cls in y_train.cpu().int().squeeze()],
        s=0.1,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("cute_graph")
    plt.show()


if __name__ == "__main__":
    _test(device=device)

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm.auto


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.weights = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(5 * 5 * 16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, X):
        return self.weights(X)


def _test():
    train_epochs = 20
    batch_size = 64
    device = "cuda"
    lr = 2e-2
    supervised_size = 150
    unsupervised = True

    model = CNN()
    model = model.to(device)

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.0,), (0.5,)),
        ]
    )

    insert_noise_low = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop((28, 28)),
            torchvision.transforms.RandomRotation(15),
        ]
    )

    insert_noise_high = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop((28, 28)),
            torchvision.transforms.RandomRotation(60),
            torchvision.transforms.GaussianBlur(7),
            torchvision.transforms.RandomAdjustSharpness(1.25),
            torchvision.transforms.RandomAutocontrast(p=0.25),
        ]
    )

    dataset_train = torchvision.datasets.MNIST(".", download=True, transform=transforms)
    (
        dataset_supervised_train,
        dataset_unsupervised_train,
    ) = torch.utils.data.random_split(
        dataset_train, [supervised_size, len(dataset_train) - supervised_size]
    )

    dataset_eval = torchvision.datasets.MNIST(
        ".", download=True, train=False, transform=transforms
    )

    dt_loader_supervised_train = torch.utils.data.DataLoader(
        dataset_supervised_train,
        batch_size=batch_size,
        shuffle=True,
    )

    sampler_unsupervised_train = torch.utils.data.DataLoader(
        dataset_unsupervised_train,
        sampler=torch.utils.data.RandomSampler(
            dataset_unsupervised_train,
            replacement=False,
        ),
        batch_size=batch_size,
    )

    dt_loader_supervised_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=4 * batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in np.arange(1, 1 + train_epochs):
        loss_train = loss_eval = 0.0
        dt_loader_unsupervised_iter = iter(sampler_unsupervised_train)

        it_train = 0
        model.train()

        for X_batch, y_batch in tqdm.auto.tqdm(dt_loader_supervised_train):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optim.zero_grad()
            y_preds = model(X_batch)
            loss = loss_supervised = criterion(y_preds, y_batch)

            if unsupervised:
                X_batch, _ = next(dt_loader_unsupervised_iter)
                X_batch_noise_low = insert_noise_low(X_batch).to(device)
                y_preds_noise_low = model(X_batch_noise_low).argmax(axis=-1)
                X_batch_noise_low = X_batch_noise_low.cpu()

                X_batch_noise_high = insert_noise_high(X_batch).to(device)
                y_preds_noise_high = model(X_batch_noise_high)
                X_batch_noise_high = X_batch_noise_high.cpu()

                loss_unsupervised = criterion(y_preds_noise_high, y_preds_noise_low)

                loss += 0.2 * loss_unsupervised

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            loss_train += loss.item()
            it_train += 1

        val_acc = 0.0
        it_eval = 0
        model.eval()

        for X_batch, y_batch in tqdm.auto.tqdm(dt_loader_supervised_eval):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_preds = model(X_batch)
            loss = criterion(y_preds, y_batch)
            loss_eval += loss.item()
            val_acc += (y_preds.argmax(axis=-1) == y_batch).float().mean().item()
            it_eval += 1

        loss_train /= it_train
        val_acc /= it_eval
        loss_eval /= it_eval

        print(f"epoch: {epoch:<11} loss train: {loss_train:.3f}")
        print(f"                   loss eval : {loss_eval:.3f}")
        print(f"                   loss acc  : {val_acc:.3f}")


if __name__ == "__main__":
    _test()

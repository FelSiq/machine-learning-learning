import typing as t

import numpy as np
import torch
import torch.nn as nn
import tqdm


class InceptionModule(nn.Module):
    def __init__(
        self,
        channels_num_out: t.Tuple[int, int, int, int] = (64, 128, 32, 32),
        channels_num_bottleneck: t.Tuple[int, int] = (96, 16),
    ):
        super(InceptionModule, self).__init__()

        in_channels = sum(channels_num_out)

        f_num1, f_num2, f_num3, f_num4 = channels_num_out
        b_num2, b_num3 = channels_num_bottleneck

        self.path1 = nn.Conv2d(in_channels, f_num1, kernel_size=1)

        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, b_num2, kernel_size=1),
            nn.Conv2d(b_num2, f_num2, kernel_size=3, padding=1),
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, b_num3, kernel_size=1),
            nn.Conv2d(b_num3, f_num3, kernel_size=5, padding=2),
        )

        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, f_num4, kernel_size=1),
        )

    def forward(self, X):
        out = torch.cat(
            (self.path1(X), self.path2(X), self.path3(X), self.path4(X)), dim=1
        )
        return out


class InceptionNet(nn.Module):
    def __init__(
        self,
        input_shape: t.Tuple[int, int, int],
        num_classes: int,
        num_layers: int,
        inception_channels_num_out: t.Tuple[int, int, int, int] = (64, 128, 32, 32),
        inception_channels_num_bottleneck: t.Tuple[int, int] = (96, 16),
    ):
        super(InceptionNet, self).__init__()

        input_c, input_h, input_w = input_shape

        inception_c = sum(inception_channels_num_out)

        self.pre_conv = nn.Conv2d(input_c, inception_c, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                InceptionModule(
                    channels_num_out=inception_channels_num_out,
                    channels_num_bottleneck=inception_channels_num_bottleneck,
                )
                for _ in range(num_layers)
            ]
        )

        self.pos_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inception_c * input_h * input_w, num_classes),
        )

    def forward(self, X):
        out = self.pre_conv(X)

        for layer in self.layers:
            out = layer(out)

        out = self.pos_net(out)

        return out


def get_data(device: str) -> t.Tuple[torch.Tensor, ...]:
    X_train = np.load("datasets/train_set_x_orig.npy") / 255.0
    X_test = np.load("datasets/test_set_x_orig.npy") / 255.0

    y_train = np.load("datasets/train_set_y_orig.npy").ravel()
    y_test = np.load("datasets/test_set_y_orig.npy").ravel()

    classes = np.load("datasets/classes.npy")

    X_train_tensor = torch.Tensor(X_train).permute((0, 3, 1, 2)).to(device)
    y_train_tensor = torch.Tensor(y_train).long().to(device)

    X_test_tensor = torch.Tensor(X_test).permute((0, 3, 1, 2))
    y_test_tensor = torch.Tensor(y_test).long()

    del X_train
    del X_test
    del y_train
    del y_test

    print(
        "X_train_tensor:", X_train_tensor.shape, "X_test_tensor:", X_test_tensor.shape
    )
    print(
        "y_train_tensor:", y_train_tensor.shape, "y_test_tensor:", y_test_tensor.shape
    )
    print("Classes:", classes)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, classes


def _test():
    import sklearn.metrics

    num_train_epochs = 10
    num_inception_layers = 2
    device = "cuda"
    checkpoint_path = f"models/inception_model_{num_inception_layers}.pt"

    X_train, X_test, y_train, y_test, classes = get_data(device)
    num_classes = len(classes)
    input_shape = X_train.shape[1:]

    full_model = InceptionNet(
        input_shape=input_shape,
        num_classes=num_classes,
        num_layers=num_inception_layers,
    )

    try:
        full_model.load_state_dict(torch.load(checkpoint_path))
        print("Checkpoint file loaded.")

    except FileNotFoundError:
        pass

    if num_train_epochs > 0:
        full_model.train()
        full_model = full_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(full_model.parameters(), lr=0.01)

        X_y_data = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(X_y_data, batch_size=32, shuffle=True)

        for i in np.arange(1, 1 + num_train_epochs):
            print(f"Epoch {i} / {num_train_epochs}...")
            for X_batch, y_batch in tqdm.auto.tqdm(dataloader):
                optim.zero_grad()
                y_preds = full_model(X_batch)
                loss = criterion(y_preds, y_batch)
                loss.backward()
                optim.step()

        torch.save(full_model.state_dict(), checkpoint_path)

    full_model.eval()  # Note: set dropout and batch normalization to eval mode

    test_preds = full_model.to("cpu")(X_test).detach().numpy().argmax(axis=1)
    test_acc = sklearn.metrics.accuracy_score(test_preds, y_test)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    _test()

import typing as t

import tqdm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernels_size: t.Sequence[int],
        filters_num: t.Sequence[t.Tuple[int, int, int]],
        strides: t.Sequence[int],
        conv_shortcut_layers: t.Set[int],
    ):
        super(ResNet, self).__init__()

        assert len(kernels_size) == len(filters_num) == len(strides)

        num_layers = len(kernels_size)

        self.layers = nn.ModuleList()

        prev_channels = in_channels

        for l in np.arange(num_layers):
            print(f"Building layer {1 + l} / {num_layers}...")
            k, f, s = kernels_size[l], filters_num[l], strides[l]

            new_block = self._build_block(
                kernel_size=k,
                filters_num=(prev_channels, *f),
                stride=s,
                conv_shortcut=l in conv_shortcut_layers,
            )

            self.layers.append(new_block)

            prev_channels = f[-1]

    @staticmethod
    def _build_block(
        kernel_size: int,
        filters_num: t.Tuple[int, int, int, int],
        stride: int = 1,
        conv_shortcut: bool = False,
    ):

        F0, F1, F2, F3 = filters_num

        # Note: make sure kernel_size is odd to keep dimensions work out as intended
        assert kernel_size % 2 == 1

        main_path = nn.Sequential(
            nn.Conv2d(
                in_channels=F0,
                out_channels=F1,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=F1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=F1,
                out_channels=F2,
                kernel_size=kernel_size,
                bias=False,
                padding=kernel_size // 2,
                padding_mode="zeros",
            ),
            nn.BatchNorm2d(num_features=F2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=F3),
        )

        if conv_shortcut:
            shortcut_path = nn.Sequential(
                nn.Conv2d(
                    in_channels=F0,
                    out_channels=F3,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(F3),
            )

        else:
            shortcut_path = nn.Identity()

        # Note: not adding 'last_activation' in 'main_path' since the skip
        # connection happens before it. Check 'forward' method.
        last_activation = nn.ReLU(inplace=True)

        return nn.ModuleList([main_path, shortcut_path, last_activation])

    def forward(self, X):
        X_out = X

        for main_path, shortcut_path, last_activation in self.layers:
            X_main_out = main_path(X_out)
            X_shortcut_out = shortcut_path(X_out)
            X_out = last_activation(X_main_out + X_shortcut_out)

        return X_out


class PreprocessNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        conv_size: int = 7,
        conv_stride: int = 2,
        pooling_size: int = 3,
        pooling_stride: int = 2,
    ):
        super(PreprocessNet, self).__init__()

        assert conv_size % 2 == 1

        self.weights = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_size,
                stride=conv_stride,
                bias=False,
                padding=conv_size // 2,
                padding_mode="zeros",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride),
        )

    def forward(self, X):
        return self.weights(X)


class PostprocessNet(nn.Module):
    def __init__(
        self,
        input_shape: t.Tuple[int, int, int],
        num_classes: int,
        pooling_size: int = 2,
    ):
        super(PostprocessNet, self).__init__()

        dense_input_size = (
            (input_shape[1] // 2) * (input_shape[2] // 2) * input_shape[0]
        )

        self.weights = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_size),
            nn.Flatten(),
            nn.Linear(dense_input_size, num_classes),
        )

    def forward(self, X):
        return self.weights(X)


class FullResNet(nn.Module):
    def __init__(
        self,
        input_shape: t.Tuple[int, int, int],
        num_classes: int,
        kernels_size: t.Union[int, t.Sequence[int]],
        filters_num: t.Sequence[t.Tuple[int, int, int]],
        strides: t.Sequence[int],
        conv_shortcut_layers: t.Set[int],
    ):
        super(FullResNet, self).__init__()

        assert len(input_shape) == 3

        input_c, input_h, input_w = input_shape

        if isinstance(kernels_size, int):
            kernels_size = len(filters_num) * [kernels_size]

        self.pre_net = PreprocessNet(in_channels=input_c)

        # Note: adjust input size to PreprocessNet output shape
        input_h = self._adjust_out_shape(input_h, 7, 2, 3)  # Conv layer
        input_h = self._adjust_out_shape(input_h, 3, 2, 0)  # Conv layer
        input_w = self._adjust_out_shape(input_w, 7, 2, 3)  # Pooling layer
        input_w = self._adjust_out_shape(input_w, 3, 2, 0)  # Pooling layer

        print("Built preprocessing layer.")

        self.mid_net = ResNet(
            in_channels=64,
            kernels_size=kernels_size,
            filters_num=filters_num,
            strides=strides,
            conv_shortcut_layers=conv_shortcut_layers,
        )

        for l in np.arange(len(filters_num)):
            # Note: only need to adjust for stride effects
            input_w = self._adjust_out_shape(input_w, 1, strides[l], 0)
            input_h = self._adjust_out_shape(input_h, 1, strides[l], 0)

        mid_net_out_shape = (filters_num[-1][-1], input_h, input_w)

        print("Input shape:", tuple(input_shape))
        print("Shape before postprocessing layer:", mid_net_out_shape)

        self.pos_net = PostprocessNet(
            input_shape=mid_net_out_shape,
            num_classes=num_classes,
        )
        print("Built postprocessing layer.")

    @staticmethod
    def _adjust_out_shape(
        input_size: int, kernel_size: int, stride: int, padding: int
    ) -> int:
        return 1 + (input_size + 2 * padding - kernel_size) // stride

    def forward(self, X):
        X_out = X

        X_out = self.pre_net(X_out)
        X_out = self.mid_net(X_out)
        X_out = self.pos_net(X_out)

        return X_out


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
    num_train_epochs = 1
    device = "cuda"

    X_train, X_test, y_train, y_test, classes = get_data(device)

    input_shape = X_train.shape[1:]
    kernels_size = 3
    filters_num = [
        (64, 64, 256),  # Conv layer
        (64, 64, 256),
        (64, 64, 256),
        (128, 128, 512),  # Conv layer
        (128, 128, 512),
        (128, 128, 512),
        (128, 128, 512),
        (256, 256, 1024),  # Conv layer
        (256, 256, 1024),
        (256, 256, 1024),
        (256, 256, 1024),
        (256, 256, 1024),
        (256, 256, 1024),
        (512, 512, 2048),  # Conv layer
        (512, 512, 2048),
        (512, 512, 2048),
    ]
    strides = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]
    conv_shortcut_layers = {0, 3, 7, 13}
    num_classes = 6

    full_model = FullResNet(
        input_shape=input_shape,
        num_classes=num_classes,
        kernels_size=kernels_size,
        filters_num=filters_num,
        strides=strides,
        conv_shortcut_layers=conv_shortcut_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(full_model.parameters(), lr=0.01)

    X_y_data = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(X_y_data, batch_size=32, shuffle=True)

    for i in np.arange(1, 1 + num_train_epochs):
        print(f"Epoch {i} / {num_train_epochs}...")
        for X_batch, y_batch in tqdm.auto.tqdm(dataloader):
            y_preds = full_model(X_batch)
            loss = criterion(y_preds, y_batch)
            loss.backward()
            optim.step()


if __name__ == "__main__":
    _test()

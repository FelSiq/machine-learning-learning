import typing as t

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

        self.layers = nn.ParameterList()

        prev_channels = in_channels

        for l in np.arange(num_layers):
            k, f, s = kernels_size[l], filters_num[l], strides[l]

            new_block = self._build_block(
                kernel_size=(prev_channels, *k),
                filters_num=f,
                stride=s,
                conv_shortcut=l in conv_shortcut_layers,
            )

            self.layers.append(new_block)

            prev_channels = k[-1]

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

        return nn.ParameterList([main_path, shortcut_path, last_activation])

    def forward(self, X):
        X_out = X

        for main_path, last_activation in self.layers:
            X_main_out = main_path(X_out)
            X_out += X_main_out  # Note: skip connection.
            X_out = last_activation(X_out)

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
        assert kernel_size % 2 == 1

        self.weights = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_size,
                stride=conv_stride,
                bias=False,
                padding=kernel_size // 2,
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
        channels_first: bool = True,
    ):
        if channels_first:
            dense_input_size = (
                (input_shape[1] // 2) * (input_shape[2] // 2) * input_shape[0]
            )

        else:
            dense_input_size = (
                (input_shape[0] // 2) * (input_shape[1] // 2) * input_shape[2]
            )

        self.weights = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_size),
            nn.Flatten(),
            nn.Dense(dense_input_size, num_classes),
        )

    def forward(self, X):
        return self.weights(X)


class FullResNet(nn.Module):
    def __init__(
        self,
        input_shape: t.Tuple[int, int, int],
        num_classes: int,
        kernels_size: t.Sequence[int],
        filters_num: t.Sequence[t.Tuple[int, int, int]],
        strides: t.Sequence[int],
        conv_shortcut_layers: t.Set[int],
        channels_first: bool = True,
    ):
        super(FullResNet, self).__init__()

        assert len(input_shape) == 3

        if channels_first:
            input_c, input_h, input_w = input_shape

        else:
            input_h, input_w, input_c = input_shape

        self.pre_net = PreprocessNet(in_channels=input_c)

        # Note: adjust input size to PreprocessNet output shape
        input_h = self._adjust_out_shape(input_h, 7, 2, 3)  # Conv layer
        input_h = self._adjust_out_shape(input_h, 3, 2, 0)  # Conv layer
        input_w = self._adjust_out_shape(input_w, 7, 2, 3)  # Pooling layer
        input_w = self._adjust_out_shape(input_w, 3, 2, 0)  # Pooling layer

        self.mid_net = ResNet(
            in_channels=input_c,
            kernel_size=kernel_size,
            filters_num=filters_num,
            strides=strides,
            conv_shortcut_layers=conv_shortcut_layers,
        )

        for l in np.arange(len(kernel_size)):
            # TODO.
            pass

        if channels_first:
            mid_net_out_shape = (kernel_size[-1][-1], input_h, input_w)

        else:
            mid_net_out_shape = (input_h, input_w, kernel_size[-1][-1])

        self.pos_net = PostprocessNet(
            input_shape=mid_net_out_shape,
            num_classes=num_classes,
            channels_first=channels_first,
        )

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


def _test():
    full_model = FullResNet()

    num_train_epochs = 32

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(full_model.parameters(), lr=0.01)

    for i in np.arange(1, 1 + num_train_epochs):
        print(f"Epoch {i} / {num_train_epochs}...")
        # TODO.
        for X_batch, y_batch in tqdm.auto.tqdm(None):
            y_preds = full_model(X_batch)
            loss = criterion(y_preds, y_batch)
            loss.backward()
            optim.step()


if __name__ == "__main__":
    _test()

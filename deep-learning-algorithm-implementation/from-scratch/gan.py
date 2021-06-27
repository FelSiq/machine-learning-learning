import typing as t

import numpy as np

import modules
import losses
import optimizers


class Generator(modules.BaseModel):
    def __init__(
        self,
        num_channels: t.Sequence[int],
        kernel_size: t.Union[int, t.Sequence[int]],
        stride: t.Union[int, t.Sequence[int]],
    ):
        super(Generator, self).__init__()

        n = len(num_channels)

        kernel_size = modules._utils.replicate(kernel_size, n - 1)
        stride = modules._utils.replicate(stride, n - 1)

        self.weights = modules.Sequential(
            [
                [
                    self._make_block(
                        channels_in=num_channels[i - 1],
                        channels_out=num_channels[i],
                        kernel_size=kernel_size[i - 1],
                        stride=stride[i - 1],
                    )
                    for i in range(1, len(num_channels) - 1)
                ],
                modules.ConvTranspose2d(
                    channels_in=num_channels[-2],
                    channels_out=num_channels[-1],
                    kernel_size=kernel_size[-1],
                    stride=stride[-1],
                ),
                modules.Tanh(),
            ]
        )

        self.weights.init_weights("normal", mean=0.0, std=0.02)

        self.noise_dim = num_channels[0]
        self.register_layers(self.weights)

    @staticmethod
    def _make_block(channels_in: int, channels_out: int, kernel_size: int, stride: int):
        ret = modules.Sequential(
            [
                modules.ConvTranspose2d(
                    channels_in=channels_in,
                    channels_out=channels_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    include_bias=False,
                ),
                modules.BatchNorm2d(channels_out),
                modules.ReLU(inplace=True),
            ]
        )

        return ret

    def forward(self, X):
        # X.shape: (batch_dim, noise_dim)
        if X.ndim != 4:
            X = np.expand_dims(X, (1, 2))

        return self.weights(X)

    def backward(self, dout):
        if dout.ndim != 4:
            dout = np.expand_dims(dout, (1, 2))

        return self.weights.backward(dout)

    def generate_noise(self, n: int = 1):
        return np.random.randn(n, 1, 1, self.noise_dim)

    def generate_and_forward(self, n: int = 1, return_noise: bool = False):
        noise = self.generate_noise(n)
        out = self.forward(noise)

        if return_noise:
            return out, noise

        return out


class Discriminator(modules.BaseModel):
    def __init__(
        self,
        num_channels: t.Sequence[int],
        kernel_size: t.Union[int, t.Sequence[int]],
        stride: t.Union[int, t.Sequence[int]],
    ):
        super(Discriminator, self).__init__()

        n = len(num_channels)

        kernel_size = modules._utils.replicate(kernel_size, n - 1)
        stride = modules._utils.replicate(stride, n - 1)

        self.weights = modules.Sequential(
            [
                [
                    self._make_block(
                        channels_in=num_channels[i - 1],
                        channels_out=num_channels[i],
                        kernel_size=kernel_size[i - 1],
                        stride=stride[i - 1],
                    )
                    for i in range(1, len(num_channels) - 1)
                ],
                modules.Conv2d(
                    channels_in=num_channels[-2],
                    channels_out=num_channels[-1],
                    kernel_size=kernel_size[-1],
                    stride=stride[-1],
                ),
            ]
        )

        self.weights.init_weights("normal", mean=0.0, std=0.02)

        self.register_layers(self.weights)

    @staticmethod
    def _make_block(channels_in: int, channels_out: int, kernel_size: int, stride: int):
        ret = modules.Sequential(
            [
                modules.Conv2d(
                    channels_in=channels_in,
                    channels_out=channels_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    include_bias=False,
                ),
                modules.BatchNorm2d(channels_out),
                modules.LeakyReLU(slope=0.2, inplace=True),
            ]
        )

        return ret

    def forward(self, X):
        # X.shape: (batch_dim, height, width, noise_dim)
        out = self.weights(X)
        return out.reshape(len(out), -1)

    def backward(self, dout):
        if dout.ndim != 4:
            dout = np.expand_dims(dout, (1, 2))

        return self.weights.backward(dout)


def _test():
    import functools
    import tqdm.auto
    import sklearn.datasets

    train_epochs = 32
    batch_size = 64

    gen = Generator(
        num_channels=[32, 32, 32, 32, 1],
        kernel_size=[2, 3, 3, 3],
        stride=1,
    )

    disc = Discriminator(
        num_channels=[1, 16, 16, 16, 32, 1],
        kernel_size=(2, 2, 2, 3, 3),
        stride=1,
    )

    print(gen.weights)
    print(disc.weights)

    X, _ = sklearn.datasets.load_digits(return_X_y=True)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X = np.nan_to_num(X, copy=False, nan=0.0)
    X = np.minimum(np.maximum(X, -2.0), 2.0)
    X = 0.5 * X
    print(X[0])

    X = X.reshape(-1, 8, 8, 1)

    optim_gen = optimizers.Nadam(gen.parameters, 2e-4, clip_grad_val=1.0)
    optim_disc = optimizers.Nadam(disc.parameters, 2e-4, clip_grad_val=1.0)

    criterion_gen = losses.BCELoss(with_logits=True)
    criterion_disc = losses.AverageLosses(
        2 * [losses.BCELoss(with_logits=True)],
        separated_y_true=True,
        separated_y_preds=True,
    )

    n = len(X)
    inds = np.arange(n)

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_gen = total_loss_disc = 0.0
        total_it = 0
        np.random.shuffle(inds)
        X = X[inds, ...]

        for start in tqdm.auto.tqdm(np.arange(0, n, batch_size)):
            end = start + batch_size
            X_batch = X[start:end, ...]

            gen.eval()
            disc.train()
            X_fake = gen.generate_and_forward(X_batch.shape[0])

            optim_disc.zero_grad()
            y_disc_real = disc(X_batch)
            y_disc_fake = disc(X_fake)

            loss_disc, (loss_grad_disc_real, loss_grad_disc_fake) = criterion_disc(
                y=[np.ones_like(y_disc_real), np.zeros_like(y_disc_fake)],
                y_preds=[y_disc_real, y_disc_fake],
            )

            disc.backward(loss_grad_disc_fake)
            disc.backward(loss_grad_disc_real)
            optim_disc.clip_grads_val()
            optim_disc.step()

            gen.train()
            disc.eval()
            optim_gen.zero_grad()
            X_fake = gen.generate_and_forward(X_batch.shape[0])
            y_disc_fake = disc(X_fake)
            loss_gen, loss_grad_gen = criterion_gen(
                y=np.ones_like(y_disc_fake), y_preds=y_disc_fake
            )
            gen.backward(loss_grad_gen)
            optim_gen.clip_grads_val()
            optim_gen.step()

            total_it += 1

            total_loss_gen += loss_gen
            total_loss_disc += loss_disc

        total_loss_gen /= total_it
        total_loss_disc /= total_it

        print(f"Avg. generator loss     : {total_loss_gen:.3f}")
        print(f"Avg. discriminator loss : {total_loss_disc:.3f}")


if __name__ == "__main__":
    _test()

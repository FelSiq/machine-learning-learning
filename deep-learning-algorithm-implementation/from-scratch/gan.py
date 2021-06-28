import typing as t

import numpy as np

import modules
import losses
import optimizers


class Generator(modules.BaseModel):
    def __init__(self, dims: t.Sequence[int]):
        super(Generator, self).__init__()

        self.weights = modules.Sequential(
            [
                [
                    self._make_block(dim_in=dims[i - 1], dim_out=dims[i])
                    for i in range(1, len(dims) - 1)
                ],
                modules.Linear(dims[-2], dims[-1]),
                modules.Tanh(),
            ]
        )

        self.noise_dim = dims[0]
        self.register_layers(self.weights)

    @staticmethod
    def _make_block(dim_in: int, dim_out: int):
        ret = modules.Sequential(
            [
                modules.Linear(dim_in, dim_out),
                # modules.BatchNorm1d(dim_out),
                modules.LeakyReLU(0.01, inplace=True),
            ]
        )

        return ret

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)

    def generate_noise(self, n: int = 1):
        return 0.5 * np.random.randn(n, self.noise_dim)

    def generate_and_forward(self, n: int = 1, return_noise: bool = False):
        noise = self.generate_noise(n)
        out = self.forward(noise)

        if return_noise:
            return out, noise

        return out


class Discriminator(modules.BaseModel):
    def __init__(self, dims: t.Sequence[int], noise_decay_iter: int = 4096):
        super(Discriminator, self).__init__()

        self.weights = modules.Sequential(
            [
                modules.AddNoiseGaussian(
                    mean=0.0,
                    std=0.050,
                    decay_std_max_iter=noise_decay_iter,
                    decay_std_min=1e-5,
                ),
                [
                    self._make_block(
                        dim_in=dims[i - 1],
                        dim_out=dims[i],
                    )
                    for i in range(1, len(dims) - 1)
                ],
                modules.Linear(dims[-2], dims[-1]),
            ]
        )

        self.register_layers(self.weights)

    @staticmethod
    def _make_block(dim_in: int, dim_out: int):
        ret = modules.Sequential(
            [
                modules.Linear(dim_in, dim_out),
                # modules.BatchNorm1d(dim_out),
                modules.LeakyReLU(slope=0.2, inplace=True),
            ]
        )

        return ret

    def forward(self, X):
        return self.weights(X)

    def backward(self, dout):
        return self.weights.backward(dout)


def _test():
    import functools
    import tqdm.auto
    import sklearn.datasets
    import matplotlib.pyplot as plt

    train_epochs = 200
    batch_size = 32

    np.random.seed(16)

    gen = Generator(dims=[16, 128, 64])

    disc = Discriminator(dims=[64, 48, 1])

    X, _ = sklearn.datasets.load_digits(return_X_y=True)
    X = (X - np.min(X, axis=0)) / (1e-7 + np.ptp(X, axis=0)) * 2.0 - 1.0

    optim_gen = optimizers.Nadam(gen.parameters, 2e-7)
    optim_disc = optimizers.SGD(disc.parameters, 2e-7)

    criterion_gen = losses.BCELoss(with_logits=True)
    criterion_disc = losses.AverageLosses(
        2 * [losses.BCELoss(with_logits=True)],
        separated_y_true=True,
        separated_y_preds=True,
    )

    n = len(X)
    inds = np.arange(n)
    freeze_disc = False

    for epoch in np.arange(1, 1 + train_epochs):
        if not freeze_disc:
            total_loss_disc = 0.0

        total_loss_gen = 0.0
        total_it = 0
        np.random.shuffle(inds)
        X = X[inds, ...]

        pgbar = tqdm.auto.tqdm(np.arange(0, n, batch_size))

        for start in pgbar:
            end = start + batch_size
            X_real = X[start:end, ...]

            if not freeze_disc:
                gen.eval()
                X_fake = gen.generate_and_forward(X_real.shape[0])

                optim_disc.zero_grad()
                y_disc_real = disc(X_real)
                y_disc_fake = disc(X_fake)

                y_true_real = (np.random.random(y_disc_real.shape) <= 0.95).astype(
                    int, copy=False
                )
                y_true_fake = (np.random.random(y_disc_fake.shape) >= 0.95).astype(
                    int, copy=False
                )

                loss_disc, (loss_grad_disc_real, loss_grad_disc_fake) = criterion_disc(
                    y=[y_true_real, y_true_fake],
                    y_preds=[y_disc_real, y_disc_fake],
                )

                disc.backward(loss_grad_disc_fake)
                disc.backward(loss_grad_disc_real)

                optim_disc.step()

            gen.train()
            optim_gen.zero_grad()
            X_fake = gen.generate_and_forward(X_real.shape[0])
            y_disc_fake = disc(X_fake)
            loss_gen, loss_grad_gen = criterion_gen(
                y=np.ones_like(y_disc_fake), y_preds=y_disc_fake
            )
            gen.backward(disc.backward(loss_grad_gen))
            optim_gen.step()

            total_it += 1

            total_loss_gen += loss_gen

            if not freeze_disc:
                total_loss_disc += loss_disc

            disc_fooled_rate = np.mean(y_disc_fake.ravel() >= 0.0)
            pgbar.set_description(
                f"{'(frozen) ' if freeze_disc else ''} Disc. fooled rate: {disc_fooled_rate:.3f}"
            )

        total_loss_gen /= total_it

        if not freeze_disc:
            total_loss_disc /= total_it

        """
        if total_loss_gen >= total_loss_disc + 0.150:
            freeze_disc = True

        if total_loss_gen <= total_loss_disc - 0.080:
            freeze_disc = False
        """

        print(f"Avg. generator loss     : {total_loss_gen:.3f}")
        print(f"Avg. discriminator loss : {total_loss_disc:.3f}")

    fig, axes = plt.subplots(2, 8, figsize=(15, 10))
    X_plot = np.squeeze(X[:8, ...])
    X_plot = 255 * (X_plot - np.min(X_plot, axis=0)) / (1e-6 + np.ptp(X_plot, axis=0))
    X_plot = X_plot.reshape(-1, 8, 8, 1)

    for i in range(8):
        axes[0][i].imshow(X_plot[i], cmap="gray", vmin=0, vmax=255)

    gen.eval()
    X_fake = np.squeeze(gen.generate_and_forward(8))
    X_fake = 255 * (X_fake - np.min(X_fake, axis=0)) / (1e-6 + np.ptp(X_fake, axis=0))
    X_fake = X_fake.reshape(-1, 8, 8, 1)

    for i in range(8):
        axes[1][i].imshow(X_fake[i], cmap="gray", vmin=0, vmax=255)

    plt.show()


if __name__ == "__main__":
    _test()

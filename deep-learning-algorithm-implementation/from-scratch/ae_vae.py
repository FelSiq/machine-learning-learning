import typing as t

import numpy as np

import modules
import optimizers
import losses


class _BaseAutoencoder(modules.BaseModel):
    def __init__(self, dims_encoder: t.Sequence[int], dims_decoder: t.Sequence[int]):
        assert len(dims_encoder) > 1
        assert len(dims_decoder) >= 1
        assert dims_encoder[0] == dims_decoder[-1]

        super(_BaseAutoencoder, self).__init__()

        self.dim_embed = int(dims_decoder[0])

        self.encoder = modules.Sequential(
            [
                [
                    self._block(dims_encoder[i - 1], dims_encoder[i])
                    for i in range(1, len(dims_encoder) - 1)
                ],
                modules.Linear(dims_encoder[-2], dims_encoder[-1]),
            ]
        )

        self.decoder = modules.Sequential(
            [
                [
                    self._block(dims_decoder[i - 1], dims_decoder[i])
                    for i in range(1, len(dims_decoder) - 1)
                ],
                modules.Linear(
                    dims_decoder[-2],
                    dims_decoder[-1],
                    activation=modules.ReLU(inplace=True),
                ),
            ]
        )

        self.register_layers(self.encoder, self.decoder)

    @staticmethod
    def _block(dim_in, dim_out):
        block = modules.Sequential(
            [
                modules.Linear(dim_in, dim_out, include_bias=False),
                modules.BatchNorm1d(dim_out),
                modules.ReLU(inplace=True),
            ]
        )

        return block

    def generate(self, n: int = 1):
        assert self.frozen
        assert int(n) > 0

        noise = np.random.randn(n, self.dim_embed)
        out = self.decoder(noise)

        return out


class Autoencoder(_BaseAutoencoder):
    def __init__(self, dims_encoder: t.Sequence[int], dims_decoder: t.Sequence[int]):
        assert dims_encoder[-1] == dims_decoder[0]
        super(Autoencoder, self).__init__(dims_encoder, dims_decoder)

    def forward(self, X):
        embed = self.encoder(X)
        out = self.decoder(embed)
        return out

    def backward(self, dout):
        dembed = self.decoder.backward(dout)
        dX = self.encoder.backward(dembed)
        return dX


class VAE(_BaseAutoencoder):
    def __init__(self, dims_encoder: t.Sequence[int], dims_decoder: t.Sequence[int]):
        super(VAE, self).__init__(dims_encoder, dims_decoder)

        self.split_mean_std = modules.Split(2, axis=1)
        self.sample_decoder_input = modules.SampleGaussianNoise()
        self.exp = modules.Exp()

        self.register_layers(
            self.split_mean_std,
            self.sample_decoder_input,
            self.exp,
        )

    def forward(self, X):
        params = self.encoder(X)
        mean, log_std = self.split_mean_std(params)
        std = self.exp(log_std)
        noise = self.sample_decoder_input(self.dim_embed, mean, std)
        out = self.decoder(noise)
        return out

    def backward(self, dout):
        dnoise = self.decoder.backward(dout)
        dmean, dlog_std = self.sample_decoder_input.backward(dnoise)
        dstd = self.exp.backward(dlog_std)
        dparams = self.split_mean_std.backward((dmean, dstd))
        dX = self.encoder.backward(dparams)
        return dX


def _test():
    import matplotlib.pyplot as plt
    import sklearn.datasets
    import tqdm.auto

    np.random.seed(32)

    eval_size = 10
    train_size = 20000
    batch_size = 32
    train_epochs = 30
    learning_rate = 1e-2

    X, _ = sklearn.datasets.load_digits(return_X_y=True)
    out_shape = (8, 8)
    np.random.shuffle(X)

    layer_dims_ae_encoder = [int(ratio * X.shape[1]) for ratio in (1.0, 0.75, 0.5)]
    layer_dims_ae_decoder = [int(ratio * X.shape[1]) for ratio in (0.5, 0.75, 1.0)]

    layer_dims_vae_encoder = [
        int(ratio * X.shape[1]) for ratio in (1.0, 0.75, 0.5, 0.2)
    ] + [2]
    layer_dims_vae_decoder = [
        int(ratio * X.shape[1]) for ratio in (0.3, 0.5, 0.75, 1.0)
    ]

    X_eval, X_train = X[:eval_size, :], X[eval_size : train_size + eval_size, :]

    X_train_max = np.max(X_train)
    X_eval /= X_train_max
    X_train /= X_train_max

    print("Train shape :", X_train.shape)
    print("Eval shape  :", X_eval.shape)

    n = X_train.shape[0]
    model_ae = Autoencoder(layer_dims_ae_encoder, layer_dims_ae_decoder)
    model_vae = VAE(layer_dims_vae_encoder, layer_dims_vae_decoder)

    criterion_ae = losses.MSELoss()
    criterion_vae = losses.MSELoss()

    optim_ae = optimizers.Nadam(
        model_ae.parameters,
        learning_rate=learning_rate,
        demon_iter_num=train_epochs / 1 * (n / batch_size),
        demon_min_mom=0.2,
    )

    optim_vae = optimizers.Nadam(
        model_vae.parameters,
        learning_rate=learning_rate,
        demon_iter_num=train_epochs / 1 * (n / batch_size),
        demon_min_mom=0.2,
    )

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_vae = total_loss_ae = 0.0
        it = 0
        np.random.shuffle(X)

        for start in tqdm.auto.tqdm(np.arange(0, n, batch_size)):
            optim_ae.zero_grad()
            optim_vae.zero_grad()

            end = start + batch_size
            X_batch = X_train[start:end, :]

            X_preds = model_ae(X_batch)
            loss_ae, loss_grad = criterion_ae(X_batch, X_preds)
            model_ae.backward(loss_grad)
            optim_ae.clip_grads_norm()
            optim_ae.step()

            X_preds = model_vae(X_batch)
            loss_vae, loss_grad = criterion_vae(X_batch, X_preds)
            model_vae.backward(loss_grad)
            optim_vae.clip_grads_norm()
            optim_vae.step()

            total_loss_ae += loss_ae
            total_loss_vae += loss_vae
            it += 1

        total_loss_ae /= it
        total_loss_vae /= it

        print(f"Total loss (AE)  : {total_loss_ae:.3f}")
        print(f"Total loss (VAE) : {total_loss_vae:.3f}")

    model_ae.eval()
    model_vae.eval()

    n = len(X_eval)

    X_preds_ae = model_ae(X_eval)
    X_preds_vae = model_vae(X_eval)

    X_gen_ae = model_ae.generate(n)
    X_gen_vae = model_vae.generate(n)

    fig, axes = plt.subplots(
        5, eval_size, figsize=(15, 10), tight_layout=True, sharex=True, sharey=True
    )

    for i in np.arange(eval_size):
        ax_orig, ax_pred_ae, ax_pred_vae, ax_gen_ae, ax_gen_vae = axes[:, i]
        ax_orig.imshow(X_eval[i, :].reshape(*out_shape))
        ax_pred_ae.imshow(X_preds_ae[i, :].reshape(*out_shape), cmap="hot")
        ax_pred_vae.imshow(X_preds_vae[i, :].reshape(*out_shape), cmap="hot")
        ax_gen_ae.imshow(X_gen_ae[i, :].reshape(*out_shape), cmap="bone")
        ax_gen_vae.imshow(X_gen_vae[i, :].reshape(*out_shape), cmap="bone")

    plt.show()


if __name__ == "__main__":
    _test()

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm.auto
import matplotlib.pyplot as plt


class SAGANBase(nn.Module):
    def __init__(
        self,
        channels_num,
        kernel_sizes,
        strides,
        num_attention_heads,
    ):
        super(SAGANBase, self).__init__()

        n = len(channels_num)

        assert n - 1 == len(kernel_sizes)
        assert n - 1 == len(strides)
        assert int(num_attention_heads) >= 0

        self.channels_num = tuple(channels_num)
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)

        nah = int(num_attention_heads)
        self.num_attention_heads = nah

        if self.num_attention_heads > 0:
            self.attention = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        chan_num,
                        num_heads=nah if chan_num % nah == 0 else 1,
                        batch_first=True,
                    )
                    for chan_num in self.channels_num[:-1]
                ]
            )
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm2d(chan_num) for chan_num in self.channels_num[:-1]]
            )

    def spatial_attention(self, conv_maps, layer_id: int):
        # conv_maps shape: (N, E, H, W)
        N, E, H, W = conv_maps.size()

        conv_maps = conv_maps.view(N, E, H * W)
        conv_maps = conv_maps.swapaxes(1, 2)

        out, _ = self.attention[layer_id](conv_maps, conv_maps, conv_maps)
        # out shape: (N, H * W, E)

        out = out.swapaxes(1, 2)
        # out shape: (N, E, H * W)

        out = out.view(N, E, H, W)

        return out

    def forward(self, X):
        out = X

        for i, layer in enumerate(self.conv_layers):
            H, W = out.size()[-2:]

            if H * W > 1 and self.num_attention_heads > 0:
                out = self.spatial_attention(out, layer_id=i) + out
                out = self.batch_norms[i](out)

            out = layer(out)

        out = self.final_layer(out)

        return out


class Generator(SAGANBase):
    def __init__(
        self,
        channels_num,
        kernel_sizes,
        strides,
        num_attention_heads: int = 8,
        conv_transpose: bool = True,
    ):
        super(Generator, self).__init__(
            channels_num=channels_num,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_attention_heads=num_attention_heads,
        )

        n = len(self.channels_num)

        self.noise_dim = self.channels_num[0]

        self.conv_layers = nn.ModuleList(
            [
                self._build_conv_block(
                    in_channels=self.channels_num[i],
                    out_channels=self.channels_num[i + 1],
                    kernel_size=self.kernel_sizes[i],
                    stride=self.strides[i],
                    conv_transpose=conv_transpose,
                )
                for i in range(n - 2)
            ]
        )

        if conv_transpose:
            self.final_layer = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(
                    nn.ConvTranspose2d(
                        in_channels=self.channels_num[-2],
                        out_channels=self.channels_num[-1],
                        kernel_size=self.kernel_sizes[-1],
                        stride=self.strides[-1],
                    )
                ),
                nn.Tanh(),
            )

        else:
            self.final_layer = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.utils.parametrizations.spectral_norm(
                    nn.Conv2d(
                        in_channels=self.channels_num[-2],
                        out_channels=self.channels_num[-1],
                        stride=self.strides[-1],
                        kernel_size=self.kernel_sizes[-1],
                    )
                ),
                nn.Tanh(),
            )

    @staticmethod
    def _build_conv_block(
        in_channels, out_channels, stride, kernel_size, conv_transpose: bool
    ):
        if conv_transpose:
            block = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        kernel_size=kernel_size,
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(0.2, inplace=False),
            )

        else:
            block = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.utils.parametrizations.spectral_norm(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        kernel_size=kernel_size,
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(0.2, inplace=True),
            )

        return block

    def gen_and_forward(self, n: int = 1, device: str = "cuda"):
        noise = 0.33 * torch.randn(n, self.noise_dim, 1, 1).to(device)
        out = self(noise)
        return out


class Discriminator(SAGANBase):
    def __init__(
        self, channels_num, kernel_sizes, strides, num_attention_heads: int = 4
    ):
        super(Discriminator, self).__init__(
            channels_num=channels_num,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_attention_heads=num_attention_heads,
        )

        n = len(self.channels_num)

        self.conv_layers = nn.ModuleList(
            [
                self._build_conv_block(
                    in_channels=self.channels_num[i],
                    out_channels=self.channels_num[i + 1],
                    kernel_size=self.kernel_sizes[i],
                    stride=self.strides[i],
                )
                for i in range(n - 2)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels_num[-2],
                out_channels=self.channels_num[-1],
                kernel_size=self.kernel_sizes[-1],
                stride=self.strides[-1],
            ),
            nn.Flatten(),
        )

    @staticmethod
    def _build_conv_block(in_channels, out_channels, stride, kernel_size):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        return block


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def _test():
    device = "cuda"
    num_epochs = 40
    batch_size = 32
    conv_transpose = True

    np.random.seed(16)
    torch.random.manual_seed(32)

    checkpoint_path = "sagan_checkpoint.tar"

    if conv_transpose:
        gen = Generator(
            channels_num=[64, 256, 128, 64, 1],
            kernel_sizes=[3, 4, 3, 4],
            strides=[1, 1, 2, 2],
            num_attention_heads=8,
            conv_transpose=conv_transpose,
        )

    else:
        gen = Generator(
            channels_num=[64, 256, 128, 128, 128, 64, 1],
            kernel_sizes=[1, 1, 3, 3, 5, 5],
            strides=[1, 1, 1, 1, 1, 1],
            num_attention_heads=8,
            conv_transpose=conv_transpose,
        )

        # aux = torch.randn(1, 32, 1, 1)
        # print(gen(aux).size())
        # exit(0)

    disc = Discriminator(
        channels_num=[1, 16, 32, 11],
        kernel_sizes=[4, 4, 4],
        strides=(2, 2, 2),
        num_attention_heads=0,
    )

    gen.apply(weights_init)
    disc.apply(weights_init)

    optim_gen = torch.optim.Adam(gen.parameters(), lr=2e-4)
    optim_disc = torch.optim.RMSprop(disc.parameters(), lr=2e-4)

    try:
        checkpoint = torch.load(checkpoint_path)
        gen.load_state_dict(checkpoint["gen"])
        gen.to(device)
        optim_gen.load_state_dict(checkpoint["optim_gen"])
        disc.load_state_dict(checkpoint["disc"])
        disc.to(device)
        optim_disc.load_state_dict(checkpoint["optim_disc"])
        print("Checkpoint loaded.")

    except FileNotFoundError:
        pass

    gen = gen.to(device)
    disc = disc.to(device)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomAffine(degrees=20, scale=(0.95, 1.05)),
            torchvision.transforms.Lambda(
                lambda x: 2.0 * (x - x.min()) / (1e-7 + x.max() - x.min()) - 1.0
            ),
        ]
    )

    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(".", download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    criterion_cls = nn.CrossEntropyLoss()

    def update_mov_avg(cur, new, it, beta=0.9):
        out = cur * beta + (1.0 - beta) * new

        if it <= 150:
            out /= 1.0 - beta ** it

        return out

    gene_it = disc_it = 0
    mv_avg_gene = mv_avg_disc = 0.0

    for epoch in np.arange(1, 1 + num_epochs):
        pbar = tqdm.auto.tqdm(dataloader)
        for i, (X_real, y_cls_label) in enumerate(pbar):
            X_real = X_real.to(device)
            y_cls_label = y_cls_label.to(device)
            batch_size = X_real.size()[0]

            optim_disc.zero_grad()
            X_fake = gen.gen_and_forward(batch_size, device).detach()
            y_preds_real = disc(X_real + 0.05 * torch.randn_like(X_real))
            y_preds_fake = disc(X_fake + 0.05 * torch.randn_like(X_fake))

            y_cls_preds = y_preds_real[:, 1:]
            y_preds_real = y_preds_real[:, 0]
            y_preds_fake = y_preds_fake[:, 0]

            y_true_real = (torch.rand_like(y_preds_real) <= 0.98).float()
            y_true_fake = (torch.rand_like(y_preds_fake) <= 0.02).float()

            y_true_real += 0.4 * torch.rand_like(y_preds_real) - 0.2
            y_true_fake += 0.4 * torch.rand_like(y_preds_fake) - 0.2

            y_true_real = torch.clip(y_true_real, 0.0, 1.0, out=y_true_real)
            y_true_fake = torch.clip(y_true_fake, 0.0, 1.0, out=y_true_fake)

            disc_loss_real = 0.7 * criterion(
                y_preds_real, y_true_real
            ) + 0.3 * criterion_cls(y_cls_preds, y_cls_label)
            disc_loss_fake = criterion(y_preds_fake, y_true_fake)
            disc_loss_total = 0.5 * (disc_loss_real + disc_loss_fake)
            disc_loss_total.backward()

            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            optim_disc.step()

            if i % 3 == 0:
                optim_gen.zero_grad()
                X_fake = gen.gen_and_forward(batch_size, device)
                y_preds = disc(X_fake)[:, 0].view(-1)
                gene_loss = criterion(y_preds, torch.ones_like(y_preds))
                gene_loss.backward()

                torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
                optim_gen.step()

                gene_it += 1
                mv_avg_gene = update_mov_avg(mv_avg_gene, gene_loss.item(), gene_it)

            disc_it += 1
            mv_avg_disc = update_mov_avg(mv_avg_disc, disc_loss_total.item(), disc_it)

            pbar.set_description(
                f"{epoch:} of {num_epochs} - gene: {mv_avg_gene:.3f}, "
                f"disc: {mv_avg_disc:.3f}"
            )

        print(f"Gene loss: {mv_avg_gene:.3f}")
        print(f"Disc loss: {mv_avg_disc:.3f}")

    checkpoint = {
        "gen": gen.state_dict(),
        "disc": disc.state_dict(),
        "optim_gen": optim_gen.state_dict(),
        "optim_disc": optim_disc.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

    gen.eval()

    n_plots = 8

    X_fake = gen.gen_and_forward(n_plots, device).cpu()
    print(X_fake.size())
    X_true, _ = next(iter(dataloader))
    X_true = X_true[:n_plots]

    X_fake = (X_fake - X_fake.min()) / (X_fake.max() - X_fake.min())
    X_true = (X_true - X_true.min()) / (X_true.max() - X_true.min())

    X_plot = torchvision.utils.make_grid(torch.cat((X_fake, X_true), dim=0), nrow=2)
    X_plot = X_plot.detach().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.imshow(X_plot)
    plt.show()


if __name__ == "__main__":
    _test()

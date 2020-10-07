import typing as t

import torch
import tqdm
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(
        self,
        num_channels: t.List[int],
        kernel_size: t.List[int],
        stride: t.List[int],
        criterion,
    ):
        super().__init__()
        self.gene = nn.Sequential(
            *(
                self._gene_block(
                    input_channels=num_channels[i],
                    output_channels=num_channels[i + 1],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                )
                for i in np.arange(num_channels.size - 2)
            ),
            self._gene_block(
                input_channels=num_channels[-2],
                output_channels=num_channels[-1],
                kernel_size=kernel_size[-1],
                stride=stride[-1],
                final_layer=True,
            ),
        )

        self.noise_dim = num_channels[0]
        self.criterion = criterion

    def _gene_block(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        final_layer: bool = False,
    ):
        if not final_layer:
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        else:
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.Tanh(),
            )

        return block

    def _unsqueeze_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return noise.view(len(noise), self.noise_dim, 1, 1)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.gene(self._unsqueeze_noise(noise))

    def get_loss(self, verdict: torch.Tensor) -> torch.Tensor:
        return self.criterion(verdict, torch.ones_like(verdict))


class Discriminator(nn.Module):
    def __init__(
        self,
        num_channels: t.List[int],
        kernel_size: t.List[int],
        stride: t.List[int],
        criterion,
    ):
        super().__init__()
        self.disc = nn.Sequential(
            *(
                self._disc_block(
                    input_channels=num_channels[i],
                    output_channels=num_channels[i + 1],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                )
                for i in np.arange(num_channels.size - 2)
            ),
            self._disc_block(
                input_channels=num_channels[-2],
                output_channels=num_channels[-1],
                kernel_size=kernel_size[-1],
                stride=stride[-1],
                final_layer=True,
            ),
        )

        self.criterion = criterion

    def _disc_block(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        negative_slope: float = 0.2,
        final_layer: bool = False,
    ):
        if not final_layer:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        else:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            )

        return block

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pred = self.disc(image)
        return pred.view(len(pred), -1)

    def get_loss(
        self, input_fake: torch.Tensor, input_real: torch.Tensor
    ) -> torch.Tensor:
        out_fake = self(input_fake)
        out_real = self(input_real)

        loss_fake = self.criterion(out_fake, torch.zeros_like(out_fake))
        loss_real = self.criterion(out_real, torch.ones_like(out_real))

        disc_loss = 0.5 * (loss_fake + loss_real)

        return disc_loss


class DCGAN:
    def __init__(
        self,
        num_channels_gene: t.List[int],
        num_channels_disc: t.List[int],
        kernel_size_gene: t.Union[t.List[int]],
        kernel_size_disc: t.Union[t.List[int]],
        stride_gene: t.Union[int, t.List[int]],
        stride_disc: t.Union[int, t.List[int]],
        lr_gene: float = 2e-4,
        lr_disc: float = 2e-4,
        device: str = "cpu",
    ):
        self.num_channels_gene = np.asarray(num_channels_gene).ravel()
        self.num_channels_disc = np.asarray(num_channels_disc).ravel()
        self.device = device
        self.lr_gene = lr_gene
        self.lr_disc = lr_disc

        if np.isscalar(kernel_size_gene):
            kernel_size_gene = np.repeat(
                kernel_size_gene, repeats=self.num_channels_gene.size - 1
            )

        if np.isscalar(stride_gene):
            stride_gene = np.repeat(
                stride_gene, repeats=self.num_channels_gene.size - 1
            )

        if np.isscalar(kernel_size_disc):
            kernel_size_disc = np.repeat(
                kernel_size_disc, repeats=self.num_channels_disc.size - 1
            )

        if np.isscalar(stride_disc):
            stride_disc = np.repeat(
                stride_disc, repeats=self.num_channels_disc.size - 1
            )

        self.kernel_size_gene = np.asarray(kernel_size_gene, dtype=int)
        self.kernel_size_disc = np.asarray(kernel_size_disc, dtype=int)
        self.stride_gene = np.asarray(stride_gene, dtype=int)
        self.stride_disc = np.asarray(stride_disc, dtype=int)

        assert np.all(
            self.kernel_size_gene > 0
        ), "All generator kernel sizes must be positive"
        assert np.all(
            self.kernel_size_disc > 0
        ), "All discriminator kernel sizes must be positive"
        assert np.all(self.stride_gene > 0), "All generator stride must be positive"
        assert np.all(self.stride_disc > 0), "All discriminator stride must be positive"
        assert device in {"cpu", "cuda"}, "'device' must be either 'cpu' or 'cuda'"
        assert self.lr_gene > 0, "'lr_gene' must be positive"
        assert self.lr_disc > 0, "'lr_disc' must be positive"
        assert (
            self.num_channels_gene.size >= 2
        ), "'num_channels_gene' have at least 2 values"
        assert (
            self.num_channels_disc.size >= 2
        ), "'num_channels_disc' have at least 2 values"
        assert (
            self.kernel_size_gene.size
            == self.num_channels_gene.size - 1
            == self.stride_gene.size
        )
        assert (
            self.kernel_size_disc.size
            == self.num_channels_disc.size - 1
            == self.stride_disc.size
        )
        assert self.num_channels_gene[-1] == self.num_channels_disc[0], (
            "Output channels of generator does not match with the input channels of"
            " discriminator"
        )

        self.noise_dim = self.num_channels_gene[0]

        self.criterion = nn.BCEWithLogitsLoss()

        self.gene = Generator(
            num_channels=self.num_channels_gene,
            kernel_size=self.kernel_size_gene,
            stride=self.stride_gene,
            criterion=self.criterion,
        ).to(device)

        self.disc = Discriminator(
            num_channels=self.num_channels_disc,
            kernel_size=self.kernel_size_disc,
            stride=self.stride_disc,
            criterion=self.criterion,
        ).to(device)

        self.gene.apply(self._weights_init)
        self.disc.apply(self._weights_init)

        self.optim_gene = torch.optim.Adam(
            self.gene.parameters(), lr=lr_gene, betas=(0.5, 0.999)
        )
        self.optim_disc = torch.optim.Adam(
            self.disc.parameters(), lr=lr_disc, betas=(0.5, 0.999)
        )

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def train(
        self, dataloader, num_epochs: int = 32, print_it: int = -1,
    ):

        if print_it < 0:
            print_it = max(1, int(0.1 * num_epochs))

        avg_loss_gene = avg_loss_disc = 0.0

        for i in np.arange(1, 1 + num_epochs):
            for input_real, _ in tqdm.auto.tqdm(dataloader):
                batch_size = len(input_real)

                input_real = input_real.to(self.device)

                input_noise = torch.randn(
                    batch_size, self.noise_dim, device=self.device
                )

                input_fake = self.gene(input_noise).detach()

                self.optim_disc.zero_grad()
                loss_disc = self.disc.get_loss(input_fake, input_real)
                loss_disc.backward(retain_graph=True)
                self.optim_disc.step()

                input_noise = torch.randn(
                    batch_size, self.noise_dim, device=self.device
                )
                input_fake = self.gene(input_noise)
                verdict = self.disc(input_fake)

                self.optim_gene.zero_grad()
                loss_gene = self.gene.get_loss(verdict)
                loss_gene.backward(retain_graph=True)
                self.optim_gene.step()

                avg_loss_gene += loss_gene.item()
                avg_loss_disc += loss_disc.item()

            if print_it > 0 and i % print_it == 0:
                avg_loss_gene /= print_it
                avg_loss_disc /= print_it

                print(36 * "-")
                print(f"Epoch: {i} of {num_epochs} ({100. * i / num_epochs:.2f}%)")
                print(f"Generator avg. loss     : {avg_loss_gene:.6f}")
                print(f"Discriminator avg. loss : {avg_loss_disc:.6f}")
                print(36 * "-", end="\n\n")

                avg_loss_gene = avg_loss_disc = 0.0

        print("Finished.")
        return self

    def generate(self, inst_num: int = 1, noise: t.Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn(inst_num, self.noise_dim, device=device)

        return self.gene.forward(noise)


def _test():
    import torchvision

    # torch.manual_seed(32)

    batch_size = 128
    num_epochs = 45

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(".", download=False, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    try:
        model = torch.load("dcgan_model.pt")

    except FileNotFoundError:
        model = DCGAN(
            num_channels_gene=[64, 256, 128, 64, 1],
            num_channels_disc=[1, 16, 32, 1],
            kernel_size_gene=[3, 4, 3, 4],
            kernel_size_disc=4,
            stride_gene=[2, 1, 2, 2],
            stride_disc=2,
            device="cuda",
        )

    try:
        model.train(dataloader=dataloader, num_epochs=num_epochs)

    except KeyboardInterrupt:
        print("Interruped.")

    print("Saving model...", end=" ")
    torch.save(model, "dcgan_model.pt")
    print("ok.")


if __name__ == "__main__":
    _test()

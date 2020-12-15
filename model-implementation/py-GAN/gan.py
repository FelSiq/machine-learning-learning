import typing as t

import torch
import tqdm
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: t.List[int], output_dim: int, criterion
    ):
        super().__init__()

        if np.isscalar(hidden_dims):
            hidden_dims = np.asarray(hidden_dims).ravel()

        self.gene = nn.Sequential(
            self._gene_block(
                input_dim, hidden_dims[0] if hidden_dims.size else output_dim
            ),
            *(
                self._gene_block(hidden_dims[i], hidden_dims[i + 1])
                for i in np.arange(hidden_dims.size - 1)
            ),
            nn.Linear(hidden_dims[-1] if hidden_dims.size else output_dim, output_dim),
            nn.Sigmoid(),
        )

        self.criterion = criterion

    def _gene_block(self, input_dim: int, output_dim: int):
        block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        return block

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.gene(noise)

    def get_loss(self, verdict: torch.Tensor) -> torch.Tensor:
        return self.criterion(verdict, torch.ones_like(verdict))


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: t.List[int], criterion):
        super().__init__()

        if np.isscalar(hidden_dims):
            hidden_dims = np.asarray(hidden_dims).ravel()

        self.disc = nn.Sequential(
            self._disc_block(input_dim, hidden_dims[0] if hidden_dims.size else 1),
            *(
                self._disc_block(hidden_dims[i], hidden_dims[i + 1])
                for i in np.arange(hidden_dims.size - 1)
            ),
            nn.Linear(hidden_dims[-1] if hidden_dims.size else input_dim, 1),
        )

        self.criterion = criterion

    def _disc_block(self, input_dim: int, output_dim: int, negative_slope: float = 0.2):
        block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        return block

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.disc(image)

    def get_loss(
        self, input_fake: torch.Tensor, input_real: torch.Tensor
    ) -> torch.Tensor:
        out_fake = self(input_fake)
        out_real = self(input_real)

        loss_fake = self.criterion(out_fake, torch.zeros_like(out_fake))
        loss_real = self.criterion(out_real, torch.ones_like(out_real))

        disc_loss = 0.5 * (loss_fake + loss_real)

        return disc_loss


class GAN:
    def __init__(
        self,
        noise_dim: int,
        hidden_dims_gen: t.List[int],
        hidden_dims_disc: t.List[int],
        gene_out_dim: int,
        lr_gene: float = 1e-5,
        lr_disc: float = 1e-5,
        device: str = "cpu",
    ):
        self.noise_dim = noise_dim
        self.hidden_dims_gen = np.asarray(hidden_dims_gen)
        self.hidden_dims_disc = np.asarray(hidden_dims_disc)
        self.genee_out_dim = gene_out_dim
        self.device = device
        self.lr_gene = lr_gene
        self.lr_disc = lr_disc

        assert noise_dim > 0, "'noise_dim' must be positive"
        assert np.all(
            self.hidden_dims_gen > 0
        ), "All generator hidden dimensions must be positive"
        assert np.all(
            self.hidden_dims_disc > 0
        ), "All discriminator hidden dimensions must be positive"
        assert self.genee_out_dim > 0, "'gene_out_dim' must be positive"
        assert device in {"cpu", "cuda"}, "'device' must be either 'cpu' or 'cuda'"
        assert self.lr_gene > 0, "'lr_gene' must be positive"
        assert self.lr_disc > 0, "'lr_disc' must be positive"

        self.criterion = nn.BCEWithLogitsLoss()

        self.gene = Generator(
            input_dim=self.noise_dim,
            hidden_dims=self.hidden_dims_gen,
            output_dim=self.genee_out_dim,
            criterion=self.criterion,
        ).to(device)

        self.disc = Discriminator(
            input_dim=self.genee_out_dim,
            hidden_dims=self.hidden_dims_disc,
            criterion=self.criterion,
        ).to(device)

        self.optim_gene = torch.optim.Adam(self.gene.parameters(), lr=lr_gene)
        self.optim_disc = torch.optim.Adam(self.disc.parameters(), lr=lr_disc)

    def train(
        self, dataloader, num_epochs: int = 32, print_it: int = -1,
    ):

        if print_it < 0:
            print_it = max(1, int(0.1 * num_epochs))

        avg_loss_gene = avg_loss_disc = 0.0

        for i in np.arange(1, 1 + num_epochs):
            for input_real, _ in tqdm.auto.tqdm(dataloader):
                batch_size = len(input_real)

                input_real = input_real.view(batch_size, -1).to(self.device)

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
            noise = torch.randn(inst_num, self.noise_dim, device=self.device)

        return self.gene.forward(noise)


def _test():
    import torchvision

    # torch.manual_seed(32)

    batch_size = 128
    num_epochs = 2

    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            ".", download=False, transform=torchvision.transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    try:
        model = torch.load("gan_model.pt")

    except FileNotFoundError:
        model = GAN(
            noise_dim=128,
            hidden_dims_gen=[128, 256, 512, 1024],
            hidden_dims_disc=[512, 256, 128],
            gene_out_dim=28 * 28,
            device="cuda",
        )

    try:
        model.train(dataloader=dataloader, num_epochs=num_epochs)

    except KeyboardInterrupt:
        print("Interruped.")

    print("Saving model...", end=" ")
    torch.save(model, "gan_model.pt")
    print("ok.")


if __name__ == "__main__":
    _test()

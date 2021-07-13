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

    def get_loss(self, crit_scores: torch.Tensor) -> torch.Tensor:
        return -torch.mean(crit_scores)


class Critic(nn.Module):
    def __init__(
        self,
        num_channels: t.List[int],
        kernel_size: t.List[int],
        stride: t.List[int],
        add_spectral_norm: bool = True,
    ):
        super().__init__()
        self.crit = nn.Sequential(
            *(
                self._crit_block(
                    input_channels=num_channels[i],
                    output_channels=num_channels[i + 1],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    add_spectral_norm=add_spectral_norm,
                )
                for i in np.arange(num_channels.size - 2)
            ),
            self._crit_block(
                input_channels=num_channels[-2],
                output_channels=num_channels[-1],
                kernel_size=kernel_size[-1],
                stride=stride[-1],
                final_layer=True,
                add_spectral_norm=add_spectral_norm,
            ),
        )

    def _crit_block(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        negative_slope: float = 0.2,
        final_layer: bool = False,
        add_spectral_norm: bool = True,
    ):
        conv_layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        if add_spectral_norm:
            conv_layer = nn.utils.spectral_norm(conv_layer)

        if not final_layer:
            block = nn.Sequential(
                conv_layer,
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        else:
            block = nn.Sequential(
                conv_layer,
            )

        return block

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pred = self.crit(image)
        return pred.view(len(pred), -1)

    def _get_gradient(
        self, input_fake: torch.Tensor, input_real: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        mixed_images = epsilon * input_real + (1 - epsilon) * input_fake
        mixed_scores = self.crit(mixed_images)

        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        return gradient

    def _get_grad_penalty(
        self, input_fake: torch.Tensor, input_real: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        gradient = self._get_gradient(
            input_fake=input_fake, input_real=input_real, epsilon=epsilon
        )
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean(torch.pow(gradient_norm - 1, 2))
        return penalty

    def get_loss(
        self,
        input_fake: torch.Tensor,
        input_real: torch.Tensor,
        epsilon: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        out_fake = self(input_fake)
        out_real = self(input_real)

        grad_penalty = self._get_grad_penalty(
            input_fake=input_fake, input_real=input_real, epsilon=epsilon
        )

        score_fake = torch.mean(out_fake)
        score_real = torch.mean(out_real)

        crit_loss = score_fake - score_real + lambda_ * grad_penalty

        return crit_loss


class WGAN:
    def __init__(
        self,
        num_channels_gene: t.List[int],
        num_channels_crit: t.List[int],
        kernel_size_gene: t.Union[t.List[int]],
        kernel_size_crit: t.Union[t.List[int]],
        stride_gene: t.Union[int, t.List[int]],
        stride_crit: t.Union[int, t.List[int]],
        lr_gene: float = 2e-4,
        lr_crit: float = 2e-4,
        device: str = "cpu",
    ):
        self.num_channels_gene = np.asarray(num_channels_gene).ravel()
        self.num_channels_crit = np.asarray(num_channels_crit).ravel()
        self.device = device
        self.lr_gene = lr_gene
        self.lr_crit = lr_crit

        if np.isscalar(kernel_size_gene):
            kernel_size_gene = np.repeat(
                kernel_size_gene, repeats=self.num_channels_gene.size - 1
            )

        if np.isscalar(stride_gene):
            stride_gene = np.repeat(
                stride_gene, repeats=self.num_channels_gene.size - 1
            )

        if np.isscalar(kernel_size_crit):
            kernel_size_crit = np.repeat(
                kernel_size_crit, repeats=self.num_channels_crit.size - 1
            )

        if np.isscalar(stride_crit):
            stride_crit = np.repeat(
                stride_crit, repeats=self.num_channels_crit.size - 1
            )

        self.kernel_size_gene = np.asarray(kernel_size_gene, dtype=int)
        self.kernel_size_crit = np.asarray(kernel_size_crit, dtype=int)
        self.stride_gene = np.asarray(stride_gene, dtype=int)
        self.stride_crit = np.asarray(stride_crit, dtype=int)

        assert np.all(
            self.kernel_size_gene > 0
        ), "All generator kernel sizes must be positive"
        assert np.all(
            self.kernel_size_crit > 0
        ), "All critic kernel sizes must be positive"
        assert np.all(self.stride_gene > 0), "All generator stride must be positive"
        assert np.all(self.stride_crit > 0), "All critic stride must be positive"
        assert device in {"cpu", "cuda"}, "'device' must be either 'cpu' or 'cuda'"
        assert self.lr_gene > 0, "'lr_gene' must be positive"
        assert self.lr_crit > 0, "'lr_crit' must be positive"
        assert (
            self.num_channels_gene.size >= 2
        ), "'num_channels_gene' have at least 2 values"
        assert (
            self.num_channels_crit.size >= 2
        ), "'num_channels_crit' have at least 2 values"
        assert (
            self.kernel_size_gene.size
            == self.num_channels_gene.size - 1
            == self.stride_gene.size
        )
        assert (
            self.kernel_size_crit.size
            == self.num_channels_crit.size - 1
            == self.stride_crit.size
        )
        assert self.num_channels_gene[-1] == self.num_channels_crit[0], (
            "Output channels of generator does not match with the input channels of"
            " critic"
        )

        self.noise_dim = self.num_channels_gene[0]

        self.gene = Generator(
            num_channels=self.num_channels_gene,
            kernel_size=self.kernel_size_gene,
            stride=self.stride_gene,
        ).to(device)

        self.crit = Critic(
            num_channels=self.num_channels_crit,
            kernel_size=self.kernel_size_crit,
            stride=self.stride_crit,
        ).to(device)

        self.gene.apply(self._weights_init)
        self.crit.apply(self._weights_init)

        self.optim_gene = torch.optim.Adam(
            self.gene.parameters(), lr=lr_gene, betas=(0.5, 0.999)
        )
        self.optim_crit = torch.optim.Adam(
            self.crit.parameters(), lr=lr_crit, betas=(0.5, 0.999)
        )

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def train(
        self,
        dataloader,
        num_epochs: int = 32,
        crit_updates: int = 5,
        print_it: int = -1,
        grad_norm_lambda: float = 10.0,
    ):

        if print_it < 0:
            print_it = max(1, int(0.1 * num_epochs))

        avg_loss_gene = avg_loss_crit = 0.0

        for i in np.arange(1, 1 + num_epochs):
            for input_real, _ in tqdm.auto.tqdm(dataloader):
                batch_size = len(input_real)
                input_real = input_real.to(self.device)

                total_loss_crit = 0.0

                for _ in np.arange(crit_updates):
                    self.optim_crit.zero_grad()

                    input_noise = torch.randn(
                        batch_size, self.noise_dim, device=self.device
                    )

                    input_fake = self.gene(input_noise).detach()

                    epsilon = torch.rand(
                        len(input_real), 1, 1, 1, device=self.device, requires_grad=True
                    )
                    loss_crit = self.crit.get_loss(
                        input_fake,
                        input_real,
                        lambda_=grad_norm_lambda,
                        epsilon=epsilon,
                    )
                    loss_crit.backward(retain_graph=True)
                    total_loss_crit += loss_crit.item()
                    self.optim_crit.step()

                self.optim_gene.zero_grad()
                input_noise = torch.randn(
                    batch_size, self.noise_dim, device=self.device
                )
                input_fake = self.gene(input_noise)
                crit_scores = self.crit(input_fake)

                loss_gene = self.gene.get_loss(crit_scores)
                loss_gene.backward(retain_graph=True)
                self.optim_gene.step()

                avg_loss_gene += loss_gene.item()
                avg_loss_crit += total_loss_crit / crit_updates

            if print_it > 0 and i % print_it == 0:
                avg_loss_gene /= print_it
                avg_loss_crit /= print_it

                print(36 * "-")
                print(f"Epoch: {i} of {num_epochs} ({100. * i / num_epochs:.2f}%)")
                print(f"Generator avg. loss : {avg_loss_gene:.6f}")
                print(f"Critic avg. loss    : {avg_loss_crit:.6f}")
                print(36 * "-", end="\n\n")

                avg_loss_gene = avg_loss_crit = 0.0

        print("Finished.")
        return self

    def generate(self, inst_num: int = 1, noise: t.Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn(inst_num, self.noise_dim, device=self.device)

        return self.gene.forward(noise)


def _test():
    import torchvision
    import matplotlib.pyplot as plt

    def _show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

    # torch.manual_seed(32)

    batch_size = 128
    num_epochs = 0
    plot = True

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

    checkpoint_file = "wgan_model.pt"

    try:
        model = torch.load(checkpoint_file)

    except FileNotFoundError:
        model = WGAN(
            num_channels_gene=[64, 256, 128, 64, 1],
            num_channels_crit=[1, 16, 32, 1],
            kernel_size_gene=[3, 4, 3, 4],
            kernel_size_crit=4,
            stride_gene=[2, 1, 2, 2],
            stride_crit=2,
            device="cuda",
        )

    if num_epochs > 0:
        try:
            model.train(dataloader=dataloader, num_epochs=num_epochs)

        except KeyboardInterrupt:
            print("Interruped.")

        print(f"Saving model to {checkpoint_file} file...", end=" ")
        torch.save(model, checkpoint_file)
        print("ok.")

    if plot:
        _show_tensor_images(model.generate(inst_num=25))


if __name__ == "__main__":
    _test()

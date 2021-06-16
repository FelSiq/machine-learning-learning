"""
Original source: https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/

My version has some differences, maily because I implemented a
Deep Restricted Botzmann Machine rather than a Shallow one.

Also I'm generating new samples from noise.
"""
import typing as t
import functools

import tqdm
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt


class RBM(nn.Module):
    def __init__(self, dims: t.Sequence[int]):
        super(RBM, self).__init__()

        self.linear = nn.ModuleList()
        self.input_bias = nn.ParameterList()
        self.hidden_bias = nn.ParameterList()

        for i in range(len(dims) - 1):
            self.linear.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            self.input_bias.append(nn.Parameter(torch.zeros(dims[i])))
            self.hidden_bias.append(nn.Parameter(torch.zeros(dims[i + 1])))

        self._last_dim = dims[-1]

    def _forward_pass(self, v, lin_layer, bias):
        prob_h = torch.sigmoid(lin_layer(v) + bias)
        return (prob_h > torch.rand_like(prob_h)).float()

    def _backward_pass(self, h, lin_layer, bias):
        prob_v = torch.sigmoid(torch.matmul(h, lin_layer.weight) + bias)
        return (prob_v > torch.rand_like(prob_v)).float()

    def forward(self, v):
        cur_v = v

        for i in range(len(self.linear)):
            cur_v = self._forward_pass(cur_v, self.linear[i], self.hidden_bias[i])

        cur_h = cur_v

        for i in reversed(range(len(self.linear))):
            cur_h = self._backward_pass(cur_h, self.linear[i], self.input_bias[i])

        cur_v = cur_h

        return cur_v

    def generate(self, n: int = 1, device: str = "cpu"):
        h = torch.rand(n, self._last_dim, device=device)
        h = h.bernoulli()
        h = h.to(device)

        cur_h = h

        for i in reversed(range(len(self.linear))):
            cur_h = self._backward_pass(cur_h, self.linear[i], self.input_bias[i])

        cur_v = cur_h

        return cur_v

    def free_energy(self, v):
        total = 0.0

        cur_v = v

        for i in range(len(self.linear)):
            bias_term = torch.matmul(cur_v, self.input_bias[i])
            next_v = self.linear[i](cur_v) + self.hidden_bias[i]
            hidden_term = next_v.exp().log1p().sum(1)
            total += -(hidden_term + bias_term).mean()
            cur_v = next_v

        return total


def _test():
    train_batch_size = 64
    eval_batch_size = 32
    train_epochs = 40
    epochs_per_plot = 2
    device = "cuda"
    dims = [28 * 28, 256, 512, 256]

    mnist = functools.partial(
        torchvision.datasets.MNIST,
        root="./",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_loader = torch.utils.data.DataLoader(
        mnist(train=True),
        batch_size=train_batch_size,
        shuffle=True,
    )

    model = RBM(dims).to(device)

    criterion = lambda x, y: model.free_energy(x) - model.free_energy(y)

    optim = torch.optim.Adam(model.parameters(), 1e-6)

    for epoch in range(1, 1 + train_epochs):
        print(f"Epoch: {epoch:4d} / {train_epochs:4d} ...")

        train_loss = 0.0
        total_batches = 0

        model.train()

        for data, _ in tqdm.auto.tqdm(train_loader):
            optim.zero_grad()

            v = nn.Parameter(data)
            v = v.view(-1, 28 * 28)
            v = v.bernoulli()  # Note: MNIST data is already normalized
            v = v.to(device)
            v_pred = model(v)

            loss = criterion(v, v_pred)
            loss.backward()
            optim.step()

            train_loss += loss.item()
            total_batches += 1

            del v

        train_loss /= total_batches
        print(f"Train loss: {train_loss:04.4f}")

        if epochs_per_plot > 0 and epoch % epochs_per_plot == 0:
            model.eval()

            v_pred = model.generate(eval_batch_size, device)
            img = v_pred.view(eval_batch_size, 1, 28, 28)
            img = torchvision.utils.make_grid(img).detach().cpu()
            img = img.permute(1, 2, 0)

            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    _test()

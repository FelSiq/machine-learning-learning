# Source: https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
import functools

import tqdm
import torch
import torch.nn as nn
import torchvision


class RBM(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, steps: int):
        super(RBM, self).__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.steps = steps

        self.linear = nn.Linear(n_input, n_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.zeros(n_input))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))
        self.sigmoid = nn.Sigmoid()

    def _forward_pass(self, v):
        prob_h = self.sigmoid(self.linear(v) + self.hidden_bias)
        return (prob_h > torch.rand_like(prob_h)).float()

    def _backward_pass(self, h):
        prob_v = self.sigmoid(torch.matmul(h, self.linear.weight) + self.input_bias)
        return (prob_v > torch.rand_like(prob_v)).float()

    def forward(self, v):
        cur_v = v

        for i in range(self.steps):
            cur_h = self._forward_pass(cur_v)
            cur_v = self._backward_pass(cur_h)

        return cur_v

    def free_energy(self, v):
        bias_term = torch.matmul(v, self.input_bias)
        input_activation = self.linear(v) + self.hidden_bias
        hidden_term = input_activation.exp().log1p().sum(1)

        return -(hidden_term + bias_term).mean()


def _test():
    train_batch_size = 64
    eval_batch_size = 128
    train_epochs = 5
    device = "cuda"

    mnist = functools.partial(
        torchvision.datasets.MNIST,
        root="./",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_loader = torch.utils.data.DataLoader(
        mnist(train=True), batch_size=train_batch_size
    )
    eval_loader = torch.utils.data.DataLoader(
        mnist(train=False), batch_size=eval_batch_size
    )

    model = RBM(n_input=28*28, n_hidden=512, steps=1).to(device)

    criterion = lambda x, y: model.free_energy(x) - model.free_energy(y)

    optim = torch.optim.SGD(model.parameters(), 0.1)

    for epoch in range(train_epochs):
        print(f"Epoch: {epoch:4d} / {train_epochs:4d} ...")

        train_loss = 0.0
        total_batches= 0

        for data, _ in tqdm.auto.tqdm(train_loader):
            optim.zero_grad()

            v = data
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


if __name__ == "__main__":
    _test()

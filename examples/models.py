import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


dense_net1 = nn.Sequential(
    Flatten(),
    nn.Linear(784, 100, bias=False),
    nn.ReLU(),
    nn.Linear(100, 10)
)

dense_net = nn.Sequential(
    Flatten(),
    nn.Linear(784, 400, bias=False),
    nn.ReLU(),
    nn.Linear(400, 100, bias=False),
    nn.ReLU(),
    nn.Linear(100, 10)
)

num_kernels = 16
kernel_size = 21
conv_net = nn.Sequential(
    nn.Conv2d(1, num_kernels, kernel_size, bias=False),
    nn.ReLU(),
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(num_kernels * int(((28 - (kernel_size - 1)) / 2)) ** 2, 10)
)

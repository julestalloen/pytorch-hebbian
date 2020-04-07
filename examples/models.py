import torch.nn as nn

from pytorch_hebbian.nn import Flatten, RePU

hidden_units = 2000
dense_net1 = nn.Sequential(
    Flatten(),
    nn.Linear(784, hidden_units, bias=False),
    RePU(1),
    nn.Linear(hidden_units, 10)
)

dense_net = nn.Sequential(
    Flatten(),
    nn.Linear(784, 400, bias=False),
    nn.ReLU(),
    nn.Linear(400, 100, bias=False),
    nn.ReLU(),
    nn.Linear(100, 10)
)

input_dim = 28
input_channels = 1

num_kernels = 8
kernel_size = 3
conv_net = nn.Sequential(
    nn.Conv2d(input_channels, num_kernels, kernel_size, bias=False),
    nn.ReLU(),
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(num_kernels * int(((input_dim - (kernel_size - 1)) / 2)) ** 2, 10)
)

num_kernels = [8, 16]
kernel_size = 3
conv_net2 = nn.Sequential(
    nn.Conv2d(input_channels, num_kernels[0], kernel_size, bias=False),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size, bias=False),
    nn.ReLU(),
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(num_kernels[1] * int(((input_dim - (kernel_size - 1)) / 2 - (kernel_size - 1)) / 2) ** 2, 10)
)

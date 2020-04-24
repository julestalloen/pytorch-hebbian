from collections import OrderedDict
from typing import List

import torch.nn as nn

from pytorch_hebbian.nn import Flatten, RePU

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


def create_fc1_model(hu: List, n: int = 1, batch_norm=False):
    modules = [
        ('flatten', Flatten()),
        ('linear1', nn.Linear(hu[0], hu[1], bias=False))
    ]

    if batch_norm:
        modules.append(('batch_norm', nn.BatchNorm1d(num_features=hu[1])))

    modules.append(('repu', RePU(n)))
    modules.append(('linear2', nn.Linear(hu[1], 10)))

    return nn.Sequential(OrderedDict(modules))

from collections import OrderedDict
from typing import List

import torch.nn as nn

from pytorch_hebbian.nn import Flatten, RePU


def create_fc1_model(hu: List, n: float = 1.0, batch_norm=False):
    modules = [
        ('flatten', Flatten()),
        ('linear1', nn.Linear(hu[0], hu[1], bias=False))
    ]

    if batch_norm:
        modules.append(('batch_norm', nn.BatchNorm1d(num_features=hu[1])))

    modules.append(('repu', RePU(n)))

    linear2 = nn.Linear(hu[1], 10)
    # nn.init.xavier_uniform_(linear2.weight.data, gain=nn.init.calculate_gain('relu'))
    modules.append(('linear2', linear2))

    return nn.Sequential(OrderedDict(modules))


def create_fc2_model():
    dense_net = nn.Sequential(
        Flatten(),
        nn.Linear(784, 400, bias=False),
        nn.ReLU(),
        nn.Linear(400, 100, bias=False),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return dense_net


def create_conv1_model(input_dim, input_channels=1, num_kernels=8, kernel_size=4, pool_size=2):
    conv_net = nn.Sequential(
        nn.Conv2d(input_channels, num_kernels, kernel_size, bias=False),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
        Flatten(),
        nn.Linear(num_kernels * int(((input_dim - (kernel_size - 1)) / 2)) ** 2, 10)
    )

    return conv_net


def create_conv2_model(input_dim, input_channels=1, num_kernels=None, kernel_size=4, pool_size=2):
    if num_kernels is None:
        num_kernels = [8, 16]

    conv_net = nn.Sequential(
        nn.Conv2d(input_channels, num_kernels[0], kernel_size, bias=False),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
        nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size, bias=False),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
        Flatten(),
        nn.Linear(num_kernels[1] * int(((input_dim - (kernel_size - 1)) / 2 - (kernel_size - 1)) / 2) ** 2, 10)
    )

    return conv_net

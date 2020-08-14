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


def create_fc2_model(hu: List, n: float = 1.0, batch_norm=False):
    modules = [
        ('flatten', Flatten()),
        ('linear1', nn.Linear(hu[0], hu[1], bias=False))
    ]

    if batch_norm:
        modules.append(('batch_norm', nn.BatchNorm1d(num_features=hu[1])))

    modules.append(('repu1', RePU(n)))

    modules.append(('linear2', nn.Linear(hu[1], hu[2], bias=False)))
    modules.append(('repu2', RePU(n)))

    linear3 = nn.Linear(hu[2], 10)
    # nn.init.xavier_uniform_(linear2.weight.data, gain=nn.init.calculate_gain('relu'))
    modules.append(('linear3', linear3))

    return nn.Sequential(OrderedDict(modules))


def create_conv1_model(input_dim, input_channels=1, num_kernels=8, kernel_size=5, pool_size=2, n=1, batch_norm=False,
                       dropout=None):
    modules = [
        ('conv1', nn.Conv2d(input_channels, num_kernels, kernel_size, bias=False))
    ]

    if batch_norm:
        modules.append(('batch_norm', nn.BatchNorm2d(num_features=num_kernels)))

    modules.extend([
        ('repu', RePU(n)),
        ('pool1', nn.MaxPool2d(pool_size)),
    ])

    if dropout is not None:
        modules.append(('dropout1', nn.Dropout2d(dropout)))

    modules.extend([
        ('flatten', Flatten()),
        ('linear1', nn.Linear(num_kernels * int(((input_dim - (kernel_size - 1)) / 2)) ** 2, 10))
    ])

    return nn.Sequential(OrderedDict(modules))


def create_conv2_model(input_dim, input_channels=1, num_kernels=None, kernel_size=4, pool_size=2, n=1):
    if num_kernels is None:
        num_kernels = [8, 16]

    modules = [
        ('conv1', nn.Conv2d(input_channels, num_kernels[0], kernel_size, bias=False)),
        ('repu1', RePU(n)),
        ('pool1', nn.MaxPool2d(pool_size)),
        ('conv2', nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size, bias=False)),
        ('repu2', RePU(n)),
        ('pool2', nn.MaxPool2d(pool_size)),
        ('flatten', Flatten()),
        ('linear1',
         nn.Linear(num_kernels[1] * int(((input_dim - (kernel_size - 1)) / 2 - (kernel_size - 1)) / 2) ** 2, 10))
    ]

    return nn.Sequential(OrderedDict(modules))

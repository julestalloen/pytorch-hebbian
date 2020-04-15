import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class RePU(nn.ReLU):
    def __init__(self, n):
        super(RePU, self).__init__()
        self.n = n

    def forward(self, x: torch.Tensor):
        return torch.relu(x) ** self.n

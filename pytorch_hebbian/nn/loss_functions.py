import torch
from torch.nn import Module


class SPELoss(Module):
    def __init__(self, m):
        super(SPELoss, self).__init__()
        self.m = m

    def forward(self, output, target):
        loss = torch.sum(torch.sum(torch.abs(output - target) ** self.m, dim=1))
        return loss

import torch
from torch.nn import Module


class SPELoss(Module):
    def __init__(self, m=1, beta=0.1):
        super(SPELoss, self).__init__()
        self.m = m
        self.beta = beta

    def forward(self, output, target):
        output = torch.tanh(self.beta * output)
        target = torch.nn.functional.one_hot(target, num_classes=output.shape[1]) * 2 - 1
        loss = torch.sum(torch.abs(output - target) ** self.m)
        return loss

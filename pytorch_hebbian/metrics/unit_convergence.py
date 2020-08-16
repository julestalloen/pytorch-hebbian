import torch
from ignite.metrics import Metric


class UnitConvergence(Metric):

    def __init__(self, layer: torch.nn.Module, norm: int, tolerance: int = 0.1, output_transform=lambda x: x,
                 device=None):
        self.layer = layer
        self.norm = norm
        self.tolerance = tolerance
        super(UnitConvergence, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        super(UnitConvergence, self).reset()

    def update(self, output):
        pass

    def compute(self):
        if type(self.layer) == torch.nn.Linear:
            weights = self.layer.weight.detach()
        elif type(self.layer) == torch.nn.Conv2d:
            weights = self.layer.weight.detach()
            weights = weights.view(-1, self.layer.kernel_size[0] * self.layer.kernel_size[1])
        else:
            raise TypeError("Layer type '{}' not supported!".format(type(self.layer)))

        sums = torch.sum(torch.pow(torch.abs(weights), self.norm), 1)
        num_converged = torch.sum(sums < (1 + self.tolerance))
        num = sums.shape[0]

        return float(num_converged) / num

import torch
from ignite.metrics import Metric


class UnitConvergence(Metric):

    def __init__(self, layer: torch.nn.Module, norm: int, tolerance: int = 0.2, output_transform=lambda x: x,
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
        weights = self.layer.weight.detach()
        sums = torch.sum(torch.pow(torch.abs(weights), self.norm), 1)
        num_converged = torch.sum(sums < (1 + self.tolerance))
        num = sums.shape[0]

        return float(num_converged) / num

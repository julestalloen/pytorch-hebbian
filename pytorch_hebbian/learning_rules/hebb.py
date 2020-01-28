import torch

from .learning_rule import LearningRule


class HebbsRule(LearningRule):

    def __init__(self, c=0.1):
        super().__init__()
        self.c = c

    def update(self, inputs, w):
        # TODO: Needs re-implementation
        d_ws = torch.zeros(inputs.size(0))
        for idx, x in enumerate(inputs):
            y = torch.dot(w, x)

            d_w = torch.zeros(w.shape)
            for i in range(y.shape[0]):
                for j in range(x.shape[0]):
                    d_w[i, j] = self.c * x[j] * y[i]

            d_ws[idx] = d_w

        return torch.mean(d_ws, dim=0)

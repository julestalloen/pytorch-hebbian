import logging

import torch

from .learning_rule import LearningRule


class KrotovsRule(LearningRule):

    def __init__(self, precision=1e-30, delta=0.4, norm=2, k=2):
        """
        Create a Krotov learning rule.
        :param precision: numerical precision of the weight updates
        :param delta: strength of the anti-hebbian learning
        :param norm: Lebesgue norm of the weights
        :param k: ranking param
        """
        super().__init__()
        self.precision = precision
        self.delta = delta
        self.norm = norm
        self.k = k

    def init_layers(self, layers: list):
        for layer in layers:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.Conv2d:
                layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, inputs, weights):
        logging.debug('Received inputs with shape {}'.format(inputs.shape))
        logging.debug('Received weights with shape {}'.format(weights.shape))

        batch_size = inputs.shape[0]
        hid = weights.shape[0]
        input_size = inputs[0].shape[0]
        inputs = torch.t(inputs)

        assert (self.k <= hid), "The amount of hidden units should be larger or equal to k!"

        sig = torch.sign(weights)
        tot_input = torch.matmul(sig * torch.abs(weights) ** (self.norm - 1), inputs)

        y = torch.argsort(tot_input, dim=0)  # fast implementation -> ranking of currents
        yl = torch.zeros((hid, batch_size))  # activations of post synaptic cells, g(Q) in [3], see also [9] and [10]
        yl[y[hid - 1, :], torch.arange(batch_size)] = 1.0  # see [10]
        yl[y[hid - self.k], torch.arange(batch_size)] = -self.delta

        xx = torch.sum(torch.mul(yl, tot_input), 1)
        # ds is the right hand side of eq [3]
        temp = torch.mul(xx.view(xx.shape[0], 1).repeat((1, input_size)), weights)
        ds = torch.matmul(yl, torch.t(inputs)) - temp

        nc = torch.max(torch.abs(ds))
        if nc < self.precision:
            nc = self.precision
        # True division
        d_w = torch.div(ds.to(dtype=torch.float), nc.to(dtype=torch.float))

        return d_w

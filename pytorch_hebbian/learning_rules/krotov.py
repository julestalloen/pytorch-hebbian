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
        for layer in [lyr.layer for lyr in layers]:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.Conv2d:
                layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, inputs: torch.Tensor, weights: torch.Tensor):
        logging.debug('Received inputs with shape {}'.format(inputs.shape))
        logging.debug('Received weights with shape {}'.format(weights.shape))

        batch_size = inputs.shape[0]
        num_hidden_units = weights.shape[0]
        input_size = inputs[0].shape[0]
        inputs = torch.t(inputs)
        assert (self.k <= num_hidden_units), "The amount of hidden units should be larger or equal to k!"

        tot_input = torch.matmul(torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), inputs)

        y = torch.argsort(tot_input, dim=0)
        activations = torch.zeros((num_hidden_units, batch_size))
        activations[y[num_hidden_units - 1, :], torch.arange(batch_size)] = 1.0
        activations[y[num_hidden_units - self.k], torch.arange(batch_size)] = -self.delta

        xx = torch.sum(torch.mul(activations, tot_input), 1)
        norm_factor = torch.mul(xx.view(xx.shape[0], 1).repeat((1, input_size)), weights)
        ds = torch.matmul(activations, torch.t(inputs)) - norm_factor

        nc = torch.max(torch.abs(ds))
        if nc < self.precision:
            nc = self.precision
        # True division
        d_w = torch.div(ds.to(dtype=torch.float), nc.to(dtype=torch.float))

        return d_w

import torch

from .learning_rule import LearningRule


class KrotovsRule(LearningRule):
    """Krotov-Hopfield Hebbian learning rule fast implementation.

    Original source: https://github.com/DimaKrotov/Biological_Learning

    Args:
        precision: Numerical precision of the weight updates.
        delta: Anti-hebbian learning strength.
        norm: Lebesgue norm of the weights.
        k: Ranking parameter
    """

    def __init__(self, precision=1e-30, delta=0.4, norm=2, k=2, normalize=False):
        super().__init__()
        self.precision = precision
        self.delta = delta
        self.norm = norm
        self.k = k
        self.normalize = normalize

    def init_layers(self, layers: list):
        for layer in [lyr.layer for lyr in layers]:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.Conv2d:
                layer.weight.data.normal_(mean=0.0, std=1.0)

    def update(self, inputs: torch.Tensor, weights: torch.Tensor):
        batch_size = inputs.shape[0]
        num_hidden_units = weights.shape[0]
        input_size = inputs[0].shape[0]
        assert (self.k <= num_hidden_units), "The amount of hidden units should be larger or equal to k!"

        # TODO: WIP
        if self.normalize:
            norm = torch.norm(inputs, dim=1)
            norm[norm == 0] = 1
            inputs = torch.div(inputs, norm.view(-1, 1))

        inputs = torch.t(inputs)

        # Calculate overlap for each hidden unit and input sample
        tot_input = torch.matmul(torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), inputs)

        # Get the top k activations for each input sample (hidden units ranked per input sample)
        _, indices = torch.topk(tot_input, k=self.k, dim=0)

        # Apply the activation function for each input sample
        activations = torch.zeros((num_hidden_units, batch_size))
        activations[indices[0], torch.arange(batch_size)] = 1.0
        activations[indices[self.k - 1], torch.arange(batch_size)] = -self.delta

        # Sum the activations for each hidden unit, the batch dimension is removed here
        xx = torch.sum(torch.mul(activations, tot_input), 1)

        # Apply the actual learning rule, from here on the tensor has the same dimension as the weights
        norm_factor = torch.mul(xx.view(xx.shape[0], 1).repeat((1, input_size)), weights)
        ds = torch.matmul(activations, torch.t(inputs)) - norm_factor

        # Normalize the weight updates so that the largest update is 1 (which is then multiplied by the learning rate)
        nc = torch.max(torch.abs(ds))
        if nc < self.precision:
            nc = self.precision
        d_w = torch.true_divide(ds, nc)

        return d_w

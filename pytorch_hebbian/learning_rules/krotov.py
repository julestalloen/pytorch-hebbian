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

    def __init__(self, precision=1e-30, delta=0.4, norm=2, k=2):
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
        self.logger.debug('Received inputs with shape {}'.format(inputs.shape))
        self.logger.debug('Received weights with shape {}'.format(weights.shape))

        batch_size = inputs.shape[0]
        num_hidden_units = weights.shape[0]
        input_size = inputs[0].shape[0]
        inputs = torch.t(inputs)
        assert (self.k <= num_hidden_units), "The amount of hidden units should be larger or equal to k!"

        # Calculate overlap with data for each hidden neuron and batch
        tot_input = torch.matmul(torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), inputs)
        self.logger.debug("tot_input shape = {}.".format(tot_input.shape))

        # Sorting the activation strengths for each batch
        y = torch.argsort(tot_input, dim=0)
        self.logger.debug("y shape = {}.".format(y.shape))

        # Activations of post-synaptic neurons for each batch
        activations = torch.zeros((num_hidden_units, batch_size))
        activations[y[num_hidden_units - 1, :], torch.arange(batch_size)] = 1.0
        activations[y[num_hidden_units - self.k], torch.arange(batch_size)] = -self.delta
        self.logger.debug("activations shape = {}.".format(activations.shape))

        # Sum the activations in each batch, the batch dimension is removed here
        xx = torch.sum(torch.mul(activations, tot_input), 1)
        self.logger.debug("xx shape = {}.".format(xx.shape))

        # Apply the actual learning rule, from here on the tensor has the same dimension as the weights
        norm_factor = torch.mul(xx.view(xx.shape[0], 1).repeat((1, input_size)), weights)
        self.logger.debug("norm_factor shape = {}.".format(norm_factor.shape))
        ds = torch.matmul(activations, torch.t(inputs)) - norm_factor
        self.logger.debug("ds shape = {}.".format(ds.shape))

        # Normalize the weight updates so that the largest update is 1 (which is then multiplied by the learning rate)
        nc = torch.max(torch.abs(ds))
        self.logger.debug("nc shape = {}.".format(nc.shape))
        if nc < self.precision:
            nc = self.precision
        d_w = torch.div(ds.to(dtype=torch.float), nc.to(dtype=torch.float))
        self.logger.debug("d_w shape = {}.".format(d_w.shape))

        return d_w

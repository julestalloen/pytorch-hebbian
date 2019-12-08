import logging

import numpy as np

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

    def update(self, inputs, w):
        logging.debug('Received inputs with shape {}'.format(inputs.shape))
        logging.debug('Received synapses with shape {}'.format(w.shape))

        synapses = w
        batch_size = inputs.shape[0]
        hid = w.shape[0]
        input_size = inputs[0].shape[0]
        inputs = np.transpose(inputs)

        assert (self.k <= hid), "The amount of hidden units should be larger or equal to k!"

        sig = np.sign(synapses)
        tot_input = np.dot(sig * np.absolute(synapses) ** (self.norm - 1), inputs)

        y = np.argsort(tot_input, axis=0)  # fast implementation -> ranking of currents
        yl = np.zeros((hid, batch_size))  # activations of post synaptic cells, g(Q) in [3], see also [9] and [10]
        yl[y[hid - 1, :], np.arange(batch_size)] = 1.0  # see [10]
        yl[y[hid - self.k], np.arange(batch_size)] = -self.delta

        xx = np.sum(np.multiply(yl, tot_input), 1)
        # ds is the right hand side of eq [3]
        temp = np.multiply(np.tile(xx.reshape(xx.shape[0], 1), (1, input_size)), synapses)
        ds = np.dot(yl, np.transpose(inputs)) - temp

        nc = np.amax(np.absolute(ds))
        if nc < self.precision:
            nc = self.precision
        d_w = np.true_divide(ds, nc)

        return d_w

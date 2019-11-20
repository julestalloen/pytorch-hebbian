import logging

from matplotlib import pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np

from pytorch_hebbian.utils.visualization import draw_weights_update


class HebbianEngine:

    def __init__(self, learning_rule, optimizer, lr_scheduler, visualize_weights: bool = False):
        self.learning_rule = learning_rule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.visualize_weights = visualize_weights

    def train(self, model: Module, data_loader: DataLoader, epochs: int):
        samples = len(data_loader.dataset)
        input_shape = tuple(next(iter(data_loader))[0].size()[2:])

        logging.info('Received {} samples with shape {}.'.format(samples, input_shape))

        # TODO: support multiple layers
        synapses = None
        for layer in model.children():
            if type(layer) == torch.nn.Linear:
                synapses = model[0].weight.detach().numpy()
                logging.info("Updating layer '{}' with shape {}.".format(layer, synapses.shape))

        # Visualization
        fig = None
        if self.visualize_weights:
            plt.ion()
            fig = plt.figure()
            draw_weights_update(fig, synapses, input_shape)

        # Main loop
        for epoch in range(epochs):
            print('epoch', epoch)
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs = np.reshape(inputs.squeeze(), (inputs.shape[0], -1))
                d_p = torch.from_numpy(self.learning_rule.update(inputs, synapses))
                self.optimizer.local_step(d_p)

            self.lr_scheduler.step()
            logging.info("Learning rate = {}.".format(self.lr_scheduler.get_lr()))

            if self.visualize_weights:
                draw_weights_update(fig, synapses, input_shape)

        # Wrap-up
        plt.ioff()
        plt.close()

        return synapses

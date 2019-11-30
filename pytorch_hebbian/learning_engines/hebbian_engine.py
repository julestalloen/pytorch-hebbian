import logging

from matplotlib import pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from pytorch_hebbian.learning_engines.learning_engine import LearningEngine
from pytorch_hebbian.utils.visualization import draw_weights_update


class HebbianEngine(LearningEngine):

    def __init__(self, learning_rule, optimizer, lr_scheduler, evaluator=None, visualize_weights: bool = False):
        super().__init__(optimizer, lr_scheduler, evaluator)
        self.learning_rule = learning_rule
        self.visualize_weights = visualize_weights

    def train(self, model: Module, data_loader: DataLoader, epochs: int,
              eval_every: int = None, checkpoint_every: int = None):
        samples = len(data_loader.dataset)
        input_shape = tuple(next(iter(data_loader))[0].size())
        _, d, h, w = input_shape
        input_shape = (h, w, d)

        logging.info('Received {} samples with shape {}.'.format(samples, input_shape))

        # TODO: support multiple layers
        weights_np = None
        for layer in list(model.children())[:-1]:
            if type(layer) == torch.nn.Linear:
                # noinspection PyUnresolvedReferences
                weights = layer.weight
                weights.data.normal_(mean=0.0, std=1.0)
                weights_np = weights.detach().numpy()
                logging.info("Updating layer '{}' with shape {}.".format(layer, weights_np.shape))

        # Visualization
        fig = None
        if self.visualize_weights:
            plt.ion()
            fig = plt.figure()
            draw_weights_update(fig, weights_np, input_shape)

        # Main loop
        for epoch in range(epochs):
            logging.info("Learning rate(s) = {}.".format(self.lr_scheduler.get_lr()))
            progress_bar = tqdm(data_loader, desc='Epoch {}/{}'.format(epoch + 1, epochs))
            for i, data in enumerate(progress_bar):
                inputs, labels = data

                labels = list(labels.numpy())
                logging.debug('Label counts: {}.'.format({label: labels.count(label) for label in np.unique(labels)}))

                inputs = np.reshape(inputs.squeeze(), (inputs.shape[0], -1))
                d_p = torch.from_numpy(self.learning_rule.update(inputs, weights_np))
                self.optimizer.local_step(d_p)

                # noinspection PyUnresolvedReferences
                weights_np = list(model.children())[0].weight.detach().numpy()

                if self.visualize_weights:
                    draw_weights_update(fig, weights_np, input_shape)

            self.lr_scheduler.step()

            if eval_every is not None:
                if epoch % eval_every == 0:
                    self.eval()

            if checkpoint_every is not None:
                if epoch % checkpoint_every == 0:
                    self.checkpoint(model)

        # Wrap-up
        plt.ioff()
        plt.close()

        return model

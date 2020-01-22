import logging

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_hebbian.learning_engines.learning_engine import LearningEngine


class HebbianEngine(LearningEngine):

    def __init__(self, learning_rule, optimizer, lr_scheduler, evaluator=None, visualizer=None):
        super().__init__(optimizer, lr_scheduler, evaluator)
        self.learning_rule = learning_rule
        self.visualizer = visualizer

    def _train_step(self, model, inputs, labels):
        labels = list(labels.numpy())
        logging.debug('Label counts: {}.'.format({label: labels.count(label) for label in np.unique(labels)}))

        inputs = np.reshape(inputs.squeeze(), (inputs.shape[0], -1))
        weights_np = list(model.children())[0].weight.detach().numpy()
        d_p = torch.from_numpy(self.learning_rule.update(inputs, weights_np))
        self.optimizer.local_step(d_p)

    def train(self, model: Module, data_loader: DataLoader, epochs: int,
              eval_every: int = None, checkpoint_every: int = None):
        # Inspect the data
        samples = len(data_loader.dataset)
        input_shape = tuple(next(iter(data_loader))[0].size())
        _, d, h, w = input_shape
        input_shape = (h, w, d)

        logging.info('Received {} samples with shape {}.'.format(samples, input_shape))

        # Iterate over the Linear layers of the model
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
        if self.visualizer is not None:
            self.visualizer.update(weights_np, input_shape)

        # Training loop
        for epoch in range(epochs):
            vis_epoch = epoch + 1
            logging.info("Learning rate(s) = {}.".format(self.lr_scheduler.get_last_lr()))

            progress_bar = tqdm(data_loader, desc='Epoch {}/{}'.format(vis_epoch, epochs))
            for inputs, labels in progress_bar:
                self._train_step(model, inputs, labels)

                # Visualization
                if self.visualizer is not None:
                    # noinspection PyUnresolvedReferences
                    weights_np = list(model.children())[0].weight.detach().numpy()
                    self.visualizer.update(weights_np, input_shape)

            self.lr_scheduler.step()

            # Evaluation
            stats = None
            if eval_every is not None:
                if vis_epoch % eval_every == 0:
                    stats = self.eval()

            # Checkpoint saving
            if checkpoint_every is not None:
                if vis_epoch % checkpoint_every == 0:
                    if stats is not None:
                        self.checkpoint(model, stats=stats)
                    else:
                        self.checkpoint(model)

        return model

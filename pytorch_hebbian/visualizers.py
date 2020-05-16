import copy
import logging
import math
import os
from abc import ABC

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from pytorch_hebbian import config, utils


class Visualizer(ABC):
    """Abstract base visualizer class to be passed to a trainer."""

    def visualize_metrics(self, metrics, epoch: int, train=False):
        pass

    def visualize_weights(self, layers: zip, input_shape, step: int):
        pass


class TensorBoardVisualizer(Visualizer):
    """TensorBoard Visualizer

    In case of blank pages (MIME type issues):
    https://github.com/tensorflow/tensorboard/issues/3117
    """

    def __init__(self, run, log_dir=config.TENSORBOARD_DIR):
        self.writer = SummaryWriter(os.path.join(log_dir, run))
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

        self.logger.info("Visualizer initiated for run '{}'.".format(run))

    def visualize_metrics(self, metrics, epoch: int, train=False):
        if train:
            mode = 'train'
        else:
            mode = 'validation'

        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        self.writer.add_scalar(mode + "/avg_loss", avg_loss, epoch)
        self.writer.add_scalar(mode + "/avg_accuracy", avg_accuracy, epoch)

    def visualize_weights(self, model: torch.nn.Module, input_shape, step: int):
        with torch.no_grad():
            first_trainable = True
            for idx, (name, layer) in enumerate(model.named_children()):
                if type(layer) == torch.nn.Linear and first_trainable:
                    # Only visualize a Linear layer if it is the first layer after the input
                    weights = layer.weight.view(-1, *input_shape)
                    first_trainable = False
                elif type(layer) == torch.nn.Conv2d:
                    weights = layer.weight
                    if input_shape[0] > 1 and idx == 0:
                        weights = weights.view(-1, input_shape[0], *weights.shape[2:])
                    else:
                        weights = weights.view(-1, 1, *weights.shape[2:])
                    first_trainable = False
                else:
                    continue

                num_weights = weights.shape[0]
                nrow = math.ceil(math.sqrt(num_weights))
                grid = torchvision.utils.make_grid(weights, nrow=nrow)

                fig = plt.figure()
                if weights.shape[1] == 1:
                    grid_np = grid[0, :].cpu().numpy()
                    nc = np.amax(np.absolute(grid_np))
                    im = plt.imshow(grid_np, cmap='bwr', vmin=-nc, vmax=nc, interpolation='nearest')
                    plt.colorbar(im, ticks=[np.amin(grid_np), 0, np.amax(grid_np)])
                else:
                    grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))
                    grid_min = np.amin(grid_np)
                    grid_max = np.amax(grid_np)
                    grid_np = (grid_np - grid_min) / (grid_max - grid_min)
                    plt.imshow(grid_np, interpolation='nearest')
                plt.axis('off')
                fig.tight_layout()

                # TODO: temp
                plt.savefig(os.path.join(config.OUTPUT_DIR, "temp-weights.png"))

                grid = utils.plot_to_img(fig)

                self.writer.add_image(name + '/weight', grid, step)

    def visualize_stats(self, model, data_loader, params):
        """Visualize the model, some input samples and the hyperparameters"""
        model = copy.deepcopy(model).cpu()
        images, labels = next(iter(data_loader))
        self.writer.add_graph(model, images)
        self.writer.add_image('input/samples', torchvision.utils.make_grid(images[:64]))
        num_project = 100
        self.project(images[:num_project], labels[:num_project])
        self.writer.add_hparams(params, {})

    def project(self, images, labels):
        input_size = torch.flatten(images[0]).size(0)
        features = images.view(-1, input_size)
        self.writer.add_embedding(features, metadata=labels, label_img=images)
        self.writer.close()

    def __del__(self):
        self.writer.close()

import math
import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import config
from pytorch_hebbian.visualizers.weights_visualizer import WeightsVisualizer


class TensorBoardVisualizer(WeightsVisualizer):
    """
    In case of blank pages (MIME type issues):
    https://github.com/tensorflow/tensorboard/issues/3117
    """
    RUNS_DIR = os.path.join(config.TENSORBOARD_DIR, 'runs')

    def __init__(self, run):
        self.writer = SummaryWriter(os.path.join(self.RUNS_DIR, run))

    def visualize_weights(self, weights, input_shape, step):
        weights = torch.reshape(weights, (-1, *input_shape))
        num_weights = weights.shape[0]
        nrow = math.ceil(math.sqrt(num_weights))
        grid = torchvision.utils.make_grid(weights, nrow=nrow)
        self.writer.add_image('weights', grid, step)

    def project(self, images, labels, input_size):
        features = images.view(-1, input_size)
        self.writer.add_embedding(features,
                                  metadata=labels,
                                  label_img=images)
        self.writer.close()

    def __del__(self):
        self.writer.close()

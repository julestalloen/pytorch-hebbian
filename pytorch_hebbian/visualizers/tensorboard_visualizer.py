import os

from torch.utils.tensorboard import SummaryWriter

import config
from pytorch_hebbian.visualizers.visualizer import Visualizer


class TensorBoardVisualizer(Visualizer):
    """
    In case of blank pages (MIME type issues):
    https://github.com/tensorflow/tensorboard/issues/3117
    """
    RUNS_DIR = os.path.join(config.TENSORBOARD_DIR, 'runs')

    def __init__(self, run='hebbian'):
        self.writer = SummaryWriter(os.path.join(self.RUNS_DIR, run))

    def update(self, weights, shape, height=None, width=None):
        pass

    def project(self, images, labels, input_size):
        features = images.view(-1, input_size)
        self.writer.add_embedding(features,
                                  metadata=labels,
                                  label_img=images)
        self.writer.close()

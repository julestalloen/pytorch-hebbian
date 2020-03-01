import logging
from abc import ABC

import torch
from ignite.contrib.handlers import LRScheduler, ProgressBar
from ignite.engine import Events
from torch.utils.data import DataLoader

from pytorch_hebbian import config
from pytorch_hebbian.visualizers import Visualizer


class Trainer(ABC):

    def __init__(self, engine, model: torch.nn.Module, lr_scheduler=None, evaluator=None,
                 visualizer: Visualizer = None):
        self.engine = engine
        self.model = model
        self.evaluator = evaluator
        self.visualizer = visualizer
        self.train_loader = None
        self.val_loader = None
        self.eval_every = None
        self.vis_weights_every = None

        if lr_scheduler is not None:
            self.lr_scheduler = LRScheduler(lr_scheduler)
            self.engine.add_event_handler(Events.EPOCH_COMPLETED, self.lr_scheduler)

        self.pbar = ProgressBar(persist=True, bar_format=config.IGNITE_BAR_FORMAT)
        self.pbar.attach(self.engine, metric_names='all')

        self._register_handlers()

    @staticmethod
    def _get_device(device):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                # Make sure all newly created tensors are cuda tensors
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                device = 'cpu'

        logging.info("Device set to '{}'.".format(device))
        return device

    def _register_handlers(self):
        if self.visualizer is not None:
            @self.engine.on(Events.ITERATION_COMPLETED)
            def visualize_weights(engine):
                if engine.state.iteration % self.vis_weights_every == 0:
                    input_shape = tuple(next(iter(self.train_loader))[0].shape[1:])
                    self.visualizer.visualize_weights(self.model, input_shape, engine.state.epoch)

    def run(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, eval_every=1, vis_weights_every=-1):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_every = eval_every

        if vis_weights_every == -1:
            self.vis_weights_every = int(len(self.train_loader.dataset) / self.train_loader.batch_size)  # every epoch
        else:
            self.vis_weights_every = vis_weights_every

        logging.info('Received {} training and {} validation samples.'.format(len(train_loader.dataset),
                                                                              len(val_loader.dataset)))
        if self.evaluator is not None:
            info_str = 'Training {} epoch(s), evaluating every {} epoch(s).'.format(epochs, self.eval_every)
        else:
            info_str = 'Training {} epoch(s).'.format(epochs)
        logging.info(info_str)
        if self.visualizer is not None:
            logging.info('Visualizing weights every {} iterations(s).'.format(self.vis_weights_every))

        self.engine.run(train_loader, max_epochs=epochs)

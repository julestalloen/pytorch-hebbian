import logging
from abc import ABC
from typing import Union, Optional

import torch
from ignite.contrib.handlers import LRScheduler, ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer
from ignite.metrics import RunningAverage
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_hebbian import utils, config
from pytorch_hebbian.learning_rules import LearningRule
from pytorch_hebbian.visualizers import Visualizer


class Trainer(ABC):
    """Abstract base trainer class.

    Supports (optional) evaluating and visualizing by default. It is up to the child class extending this class to
    register the handler for evaluation.
    """

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
            @self.engine.on(Events.STARTED)
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


class SupervisedTrainer(Trainer):
    """Trains a model using classical supervised backpropagation."""

    def __init__(self, model: torch.nn.Module, optimizer: Optimizer, criterion, evaluator, lr_scheduler=None,
                 visualizer: Visualizer = None, device: Optional[Union[str, torch.device]] = None):
        device = self._get_device(device)
        engine = create_supervised_trainer(model, optimizer, criterion, device=device)

        # TODO: only works if the evaluator class takes model and criterion as input!
        self.train_evaluator = evaluator.__class__(model, criterion)
        RunningAverage(output_transform=lambda x: x).attach(engine, 'loss')

        super().__init__(engine, model, lr_scheduler, evaluator, visualizer)

    def _register_handlers(self):
        super()._register_handlers()

        if self.train_evaluator is not None:
            @self.engine.on(Events.EPOCH_COMPLETED)
            def log_training_results(engine):
                if engine.state.epoch % self.eval_every == 0:
                    self.train_evaluator.run(self.train_loader)
                    metrics = self.train_evaluator.engine.state.metrics
                    avg_accuracy = metrics['accuracy']
                    avg_loss = metrics['loss']

                    self.pbar.log_message(config.TRAIN_REPORT_FORMAT.format(engine.state.epoch, avg_accuracy, avg_loss))

                    if self.visualizer is not None:
                        self.visualizer.visualize_metrics(metrics, engine.state.epoch, train=True)

        if self.evaluator is not None:
            @self.engine.on(Events.EPOCH_COMPLETED)
            def log_validation_results(engine):
                if engine.state.epoch % self.eval_every == 0:
                    self.evaluator.run(self.val_loader)
                    metrics = self.evaluator.engine.state.metrics
                    avg_accuracy = metrics['accuracy']
                    avg_loss = metrics['loss']

                    self.pbar.log_message(config.EVAL_REPORT_FORMAT.format(engine.state.epoch, avg_accuracy, avg_loss))
                    self.pbar.n = self.pbar.last_print_n = 0

                    if self.visualizer is not None:
                        self.visualizer.visualize_metrics(metrics, engine.state.epoch)


class HebbianTrainer(Trainer):
    """Trains a model using unsupervised local learning rules also known as Hebbian learning."""

    def __init__(self, model: torch.nn.Sequential, learning_rule: LearningRule, optimizer: Optimizer, lr_scheduler,
                 evaluator=None, supervised_from=-1, visualizer: Visualizer = None,
                 device: Optional[Union[str, torch.device]] = None):
        device = self._get_device(device)
        engine = self.create_hebbian_trainer(model, learning_rule, optimizer, device)
        self.supervised_from = supervised_from

        # Get the Hebbian trainable layers
        self.layers = []
        self.layer_indices = []
        self.layer_names = []
        for idx, named_param in enumerate(list(model.named_children())[:self.supervised_from]):
            name, layer = named_param
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.Conv2d:
                self.layers.append(layer)
                self.layer_indices.append(idx)
                self.layer_names.append(name)

        logging.info("Received {} trainable layer(s): {}.".format(len(self.layers), self.layer_names))

        # Initialize layer weights according to the learning rule
        self.learning_rule = learning_rule
        self.learning_rule.init_layers(self.layers)

        super().__init__(engine, model, lr_scheduler, evaluator, visualizer)

    def _register_handlers(self):
        super()._register_handlers()

        if self.evaluator is not None:
            @self.engine.on(Events.EPOCH_COMPLETED)
            def log_validation_results(engine):
                if engine.state.epoch % self.eval_every == 0:
                    self.evaluator.run(self.train_loader, self.val_loader, supervised_from=self.supervised_from)
                    metrics = self.evaluator.metrics
                    avg_accuracy = metrics['accuracy']
                    avg_loss = metrics['loss']

                    self.pbar.log_message(config.EVAL_REPORT_FORMAT.format(engine.state.epoch, avg_accuracy, avg_loss))
                    self.pbar.n = self.pbar.last_print_n = 0

                    if self.visualizer is not None:
                        self.visualizer.visualize_metrics(metrics, engine.state.epoch)

    @staticmethod
    def _prepare_data(inputs, model, layer_index):
        """Prepare the inputs and layer weights to be passed to the learning rule."""
        layers = list(model.children())
        layer = layers[layer_index]

        # Get the input right before the layer
        if layer_index == 0:
            x = inputs
        else:
            x = inputs
            for lyr in layers[:layer_index]:
                x = lyr(x)

        # Get the layer weight and input image patches
        if type(layer) == torch.nn.Linear:
            w = layer.weight
        elif type(layer) == torch.nn.Conv2d:
            w = layer.weight
            w = w.view(-1, layer.kernel_size[0] * layer.kernel_size[1])
            x = utils.extract_image_patches(x, kernel_size=layer.kernel_size, stride=layer.stride,
                                            padding=layer.padding, dilation=layer.dilation)
        else:
            raise TypeError("Unsupported layer type!")

        x = x.view((x.shape[0], -1))
        logging.debug("Prepared inputs and weights with shapes {} and {}.".format(list(x.shape), list(w.shape)))
        return x, w

    def create_hebbian_trainer(self, model: torch.nn.Module, learning_rule, optimizer, device=None, non_blocking=False,
                               prepare_batch=utils.prepare_batch,
                               output_transform=lambda x, y: 0):
        def _update(_, batch):
            # TODO: should this be .train() or .eval()?
            model.train()
            with torch.no_grad():
                x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

                # Train layer per layer
                for layer_index, layer_name, layer in zip(self.layer_indices, self.layer_names, self.layers):
                    logging.debug("Updating layer '{}' with shape {}.".format(layer, layer.weight.shape))
                    inputs, weights = self._prepare_data(x, model, layer_index)
                    d_p = learning_rule.update(inputs, weights)
                    d_p = d_p.view(*layer.weight.size())
                    optimizer.local_step(d_p, layer_name=layer_name)

                return output_transform(x, y)

        return Engine(_update)

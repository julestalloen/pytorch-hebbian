import logging
import math
from abc import ABC
from collections import namedtuple
from typing import Union, Optional, Dict, List, Callable, Sequence

import torch
from ignite.contrib.handlers import ProgressBar, global_step_from_engine
from ignite.engine import Engine, Events, create_supervised_trainer
from ignite.metrics import RunningAverage
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_hebbian import utils, config
from pytorch_hebbian.handlers.tqdm_logger import TqdmLogger, OutputHandler
from pytorch_hebbian.learning_rules import LearningRule
from pytorch_hebbian.visualizers import Visualizer


class Trainer(ABC):
    """Abstract base trainer class.

    Supports (optional) evaluating and visualizing by default.
    """

    def __init__(self, engine, model: torch.nn.Module, evaluator=None, train_evaluator=None,
                 evaluator_args: Callable[[], dict] = None, visualizer: Visualizer = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.engine = engine
        self.model = model
        self.evaluator = evaluator
        self.train_evaluator = train_evaluator
        self.visualizer = visualizer
        self.device = utils.get_device(device)
        self.train_loader = None
        self.val_loader = None
        self.eval_every = None
        self.vis_weights_every = None
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

        # Move the model to the appropriate device
        self.model.to(device)

        if evaluator_args is None:
            self.evaluator_args = lambda: {'val_loader': self.val_loader}
        else:
            self.evaluator_args = evaluator_args

        self.pbar = ProgressBar(persist=True, bar_format=config.IGNITE_BAR_FORMAT)
        self.pbar.attach(self.engine, metric_names='all')
        self.tqdm_logger = TqdmLogger(pbar=self.pbar)

        self._register_handlers()

    def _register_handlers(self):
        if self.visualizer is not None:
            @self.engine.on(Events.STARTED)
            @self.engine.on(Events.ITERATION_COMPLETED)
            def visualize_weights(engine):
                if engine.state.iteration % self.vis_weights_every == 0:
                    input_shape = tuple(next(iter(self.train_loader))[0].shape[1:])
                    self.visualizer.visualize_weights(self.model, input_shape, engine.state.epoch)

        if self.train_evaluator is not None:
            self.tqdm_logger.attach(self.train_evaluator.engine,
                                    log_handler=OutputHandler(tag="train",
                                                              global_step_transform=global_step_from_engine(
                                                                  self.engine)),
                                    event_name=Events.COMPLETED)

            @self.engine.on(Events.EPOCH_COMPLETED)
            def log_training_results(engine):
                if engine.state.epoch % self.eval_every == 0:
                    self.train_evaluator.run(self.train_loader)

                    if self.visualizer is not None:
                        self.visualizer.visualize_metrics(self.train_evaluator.engine.state.metrics,
                                                          engine.state.epoch, train=True)

        if self.evaluator is not None:
            self.tqdm_logger.attach(self.evaluator.engine,
                                    log_handler=OutputHandler(tag="validation",
                                                              global_step_transform=global_step_from_engine(
                                                                  self.engine)),
                                    event_name=Events.COMPLETED)

            @self.engine.on(Events.EPOCH_COMPLETED)
            def log_validation_results(engine):
                if engine.state.epoch % self.eval_every == 0:
                    self.evaluator.run(**self.evaluator_args())

                    if self.visualizer is not None:
                        self.visualizer.visualize_metrics(self.evaluator.engine.state.metrics, engine.state.epoch)

    def run(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 10, eval_every=1,
            vis_weights_every=-1):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_every = eval_every

        if vis_weights_every == -1:
            self.vis_weights_every = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
        else:
            self.vis_weights_every = vis_weights_every

        self.logger.info('Received {} training samples.'.format(len(train_loader.dataset)))
        self.logger.info('Training {} epoch(s).'.format(epochs))

        self.engine.run(train_loader, max_epochs=epochs)


class SupervisedTrainer(Trainer):
    """Trains a model using classical supervised backpropagation.

    Args:
        model: The model to be trained.
        optimizer: The optimizer used to train the model.
        criterion: The criterion used for calculating the loss.
        evaluator: An optional evaluator.
        visualizer: An optional visualizer.
        device: The device to be used.
    """

    def __init__(self, model: torch.nn.Module, optimizer: Optimizer, criterion, train_evaluator=None, evaluator=None,
                 visualizer: Visualizer = None, device: Optional[Union[str, torch.device]] = None):
        device = utils.get_device(device)
        engine = create_supervised_trainer(model, optimizer, criterion, device=device)
        RunningAverage(output_transform=lambda x: x).attach(engine, 'loss')

        super().__init__(engine=engine, model=model, evaluator=evaluator, train_evaluator=train_evaluator,
                         visualizer=visualizer, device=device)


class HebbianTrainer(Trainer):
    """Trains a model using unsupervised local learning rules also known as Hebbian learning.

    The specified learning rule is used to perform local weight updates after each batch of data. Per batch all
    trainable layers are updated in sequence.

    Args:
        model (torch.nn.Sequential): The model to be trained.
        learning_rule (Union[LearningRule, Dict[str, LearningRule]]):
            The learning rule(s) used to update the model weights.
        optimizer (Optimizer): The optimizer used to perform the weight updates.
        evaluator: An optional evaluator.
        supervised_from (int): From which layer (name) the training should be performed supervised.
        freeze_layers (list): Layers (names) to freeze during training.
        visualizer (Visualizer): An optional visualizer.
        device (Optional[Union[str, torch.device]]): The device to perform the training on.

    Attributes:
        supervised_from: See the supervised_from arg.
        freeze_layers: See the freeze_layers arg.
        layers: The Hebbian trainable layers.
    """

    def __init__(self, model: torch.nn.Sequential, learning_rule: Union[LearningRule, Dict[str, LearningRule]],
                 optimizer: Optimizer, evaluator=None, supervised_from: int = -1,
                 freeze_layers: List[str] = None, visualizer: Visualizer = None,
                 device: Optional[Union[str, torch.device]] = None):
        device = utils.get_device(device)
        engine = self.create_hebbian_trainer(model, learning_rule, optimizer, device=device)
        self.supervised_from = supervised_from
        self.freeze_layers = freeze_layers
        if self.freeze_layers is None:
            self.freeze_layers = []

        # Get the Hebbian trainable layers
        Layer = namedtuple('Layer', ['idx', 'name', 'layer'])
        self.layers = []
        for idx, (name, layer) in enumerate(list(model.named_children())[:self.supervised_from]):
            if (type(layer) == torch.nn.Linear or type(layer) == torch.nn.Conv2d) and name not in self.freeze_layers:
                self.layers.append(Layer(idx, name, layer))

        # Initialize layer weights according to the learning rule
        self.learning_rule = learning_rule
        if type(self.learning_rule) == dict:
            for rule in self.learning_rule.values():
                rule.init_layers(self.layers)
        else:
            self.learning_rule.init_layers(self.layers)

        super().__init__(engine=engine, model=model, evaluator=evaluator, visualizer=visualizer, device=device,
                         evaluator_args=lambda: {
                             'train_loader': self.train_loader,
                             'val_loader': self.val_loader,
                             'supervised_from': self.supervised_from,
                         })

        self.logger.info(
            "Received {} trainable layer(s): {}.".format(len(self.layers), [lyr.name for lyr in self.layers]))

    def _prepare_data(self, inputs, model, layer_index):
        """Prepare the inputs and layer weights to be passed to the learning rule.

        Args:
            inputs: The input to the model.
            model: The model to be trained.
            layer_index: The index of the layer currently being trained.
        """
        layers = list(model.children())
        layer = layers[layer_index]

        # Get the input right before the layer
        if layer_index == 0:
            x = inputs
        else:
            x = inputs
            for lyr in layers[:layer_index]:
                x = lyr(x)

        # TODO: trigger forward hooks for remaining layers
        model(inputs)

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
        self.logger.debug("Prepared inputs and weights with shapes {} and {}.".format(list(x.shape), list(w.shape)))
        return x, w

    def create_hebbian_trainer(self, model: torch.nn.Module, learning_rule, optimizer, device=None, non_blocking=False,
                               prepare_batch=utils.prepare_batch,
                               output_transform=lambda x, y: 0):
        def _update(_, batch: Sequence[torch.Tensor]):
            model.train()
            with torch.no_grad():
                x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

                # Train layer per layer
                for layer_index, layer_name, layer in self.layers:
                    self.logger.debug("Updating layer '{}' with shape {}.".format(layer, layer.weight.shape))
                    inputs, weights = self._prepare_data(x, model, layer_index)

                    if type(learning_rule) == dict:
                        try:
                            rule = learning_rule[layer_name]
                        except KeyError:
                            self.logger.error("No learning rule was specified for layer '{}'!".format(layer_name))
                            raise
                    else:
                        rule = learning_rule

                    d_p = rule.update(inputs, weights)
                    d_p = d_p.view(*layer.weight.size())
                    optimizer.local_step(d_p, layer_name=layer_name)

                return output_transform(x, y)

        return Engine(_update)

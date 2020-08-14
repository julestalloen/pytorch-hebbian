import logging
from abc import ABC
from collections import namedtuple
from functools import partial
from typing import Union, Optional, Dict, List, Sequence

import torch
from ignite.engine import Engine, create_supervised_trainer
from ignite.metrics import RunningAverage
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_hebbian import utils
from pytorch_hebbian.learning_rules import LearningRule


class Trainer(ABC):
    """Abstract base trainer class.

    Supports (optional) evaluating and visualizing by default.
    """

    def __init__(self, engine, model: torch.nn.Module, device: Optional[Union[str, torch.device]] = None):
        self.engine = engine
        self.model = model
        self.device = utils.get_device(device)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def run(self, train_loader: DataLoader, epochs: int = 10):
        self.engine.run(train_loader, max_epochs=epochs)


class SupervisedTrainer(Trainer):
    """Trains a model using classical supervised backpropagation.

    Args:
        model: The model to be trained.
        optimizer: The optimizer used to train the model.
        criterion: The criterion used for calculating the loss.
        device: The device to be used.
    """

    def __init__(self, model: torch.nn.Module, optimizer: Optimizer, criterion,
                 device: Optional[Union[str, torch.device]] = None):
        device = utils.get_device(device)
        engine = create_supervised_trainer(model, optimizer, criterion, device=device)
        RunningAverage(output_transform=lambda x: x).attach(engine, 'loss')

        super().__init__(engine=engine, model=model, device=device)


class HebbianTrainer(Trainer):
    """Trains a model using unsupervised local learning rules also known as Hebbian learning.

    The specified learning rule is used to perform local weight updates after each batch of data. Per batch all
    trainable layers are updated in sequence.

    Args:
        model (torch.nn.Sequential): The model to be trained.
        learning_rule (LearningRule | Dict[str, LearningRule]):
            The learning rule(s) used to update the model weights.
        optimizer (Optimizer): The optimizer used to perform the weight updates.
        supervised_from (int): From which layer (name) the training should be performed supervised.
        freeze_layers (list): Layers (names) to freeze during training.
        device (Optional[Union[str, torch.device]]): The device to perform the training on.

    Attributes:
        supervised_from: See the supervised_from arg.
        freeze_layers: See the freeze_layers arg.
        layers: The Hebbian trainable layers.
    """

    def __init__(self, model: torch.nn.Sequential, learning_rule: Union[LearningRule, Dict[str, LearningRule]],
                 optimizer: Optimizer, supervised_from: int = -1, freeze_layers: List[str] = None,
                 complete_forward: bool = False, single_forward: bool = False,
                 device: Optional[Union[str, torch.device]] = None):
        device = utils.get_device(device)
        engine = self.create_hebbian_trainer(model, learning_rule, optimizer, device=device)
        self.supervised_from = supervised_from
        self.freeze_layers = freeze_layers
        self.complete_forward = complete_forward
        self.single_forward = single_forward
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

        super().__init__(engine=engine, model=model, device=device)

        self.logger.info(
            "Received {} trainable layer(s): {}.".format(len(self.layers), [lyr.name for lyr in self.layers]))

        if self.single_forward:
            # Register hooks to store trainable layer outputs
            self._hooks = {}
            self._inputs = {}
            self._outputs = {}
            for lyr in self.layers:
                self._hooks[lyr.name] = lyr.layer.register_forward_hook(
                    partial(self._store_data_hook, layer_name=lyr.name))

    def _store_data_hook(self, _, inp, output, layer_name):
        self._inputs[layer_name] = inp[0]
        self._outputs[layer_name] = output

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

        if self.complete_forward:
            for lyr in layers[layer_index:]:
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
        self.logger.debug("Prepared inputs and weights with shapes {} and {}.".format(list(x.shape), list(w.shape)))
        return x, w

    def _prepare_data2(self, layer, layer_name):
        x = self._inputs[layer_name]
        y = self._outputs[layer_name]

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
        return x, y, w

    def _forward(self, inputs, model):
        if self.complete_forward:
            model(inputs)
        else:
            layers = list(model.children())
            x = inputs
            for lyr in layers[:self.supervised_from - 1]:
                x = lyr(x)

    def create_hebbian_trainer(self, model: torch.nn.Module, learning_rule, optimizer, device=None, non_blocking=False,
                               prepare_batch=utils.prepare_batch,
                               output_transform=lambda x, y: 0):
        def _update(_, batch: Sequence[torch.Tensor]):
            model.train()
            with torch.no_grad():
                x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
                if self.single_forward:
                    self._forward(x, model)

                # Train layer per layer
                for layer_index, layer_name, layer in self.layers:
                    self.logger.debug("Updating layer '{}' with shape {}.".format(layer, layer.weight.shape))
                    if self.single_forward:
                        inputs, _, weights = self._prepare_data2(layer, layer_name)
                    else:
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

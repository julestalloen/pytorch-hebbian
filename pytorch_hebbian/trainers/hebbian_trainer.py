import logging
from typing import Union, Optional

import torch
from ignite.engine import Engine, Events
from torch.optim.optimizer import Optimizer

from pytorch_hebbian import utils, config
from pytorch_hebbian.learning_rules import LearningRule
from pytorch_hebbian.trainers import Trainer
from pytorch_hebbian.visualizers import Visualizer


class HebbianTrainer(Trainer):
    """Trains a model using unsupervised local learning rules also known as Hebbian learning."""

    def __init__(self, model: torch.nn.Sequential, learning_rule: LearningRule, optimizer: Optimizer, lr_scheduler,
                 evaluator=None, visualizer: Visualizer = None, device: Optional[Union[str, torch.device]] = None):
        device = self._get_device(device)
        engine = self.create_hebbian_trainer(model, learning_rule, optimizer, device)

        # Get the Hebbian trainable layers
        self.layers = []
        self.layer_indices = []
        self.layer_names = []
        for idx, named_param in enumerate(list(model.named_children())[:-1]):
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
                    self.evaluator.run(self.train_loader, self.val_loader)
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
            w = w.view(w.size(0), -1)
            x = utils.image.extract_image_patches(x, kernel_size=layer.kernel_size, stride=layer.stride,
                                                  padding=layer.padding, dilation=layer.dilation)
        else:
            raise TypeError("Unsupported layer type!")

        x = x.view((x.shape[0], -1))
        logging.debug("Prepared inputs and weights with shapes {} and {}.".format(list(x.shape), list(w.shape)))
        return x, w

    def create_hebbian_trainer(self, model: torch.nn.Module, learning_rule, optimizer, device=None, non_blocking=False,
                               prepare_batch=utils.data.prepare_batch,
                               output_transform=lambda x, y: 0):
        def _update(_, batch):
            # TODO: should this be .train() or .eval()?
            model.train()
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

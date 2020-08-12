import math
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torchvision
from ignite.contrib.handlers.base_logger import BaseHandler, BaseWeightsScalarHandler, BaseWeightsHistHandler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from matplotlib import pyplot as plt

__all__ = [
    'WeightsScalarHandler',
    'WeightsHistHandler',
    'NumActivationsScalarHandler',
    'ActivationsScalarHandler',
    'ActivationsHistHandler',
    'WeightsImageHandler',
]


class WeightsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's weights as scalars.
    Handler iterates over named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, 'generator'
    """

    def __init__(self, model, reduction=torch.norm, layer_names=None, tag=None):
        super(WeightsScalarHandler, self).__init__(model, reduction, tag=tag)
        self.layer_names = layer_names

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if self.layer_names is not None:
                if name.split('.')[0] not in self.layer_names:
                    continue

            name = name.replace(".", "/")
            logger.writer.add_scalar(
                "{}weights_{}/{}".format(tag_prefix, self.reduction.__name__, name), self.reduction(p.data), global_step
            )


class WeightsHistHandler(BaseWeightsHistHandler):
    """Helper handler to log model's weights as histograms.

    Args:
        model (torch.nn.Module): model to log weights
        tag (str, optional): common title for all produced plots. For example, 'generator'
    """

    def __init__(self, model, layer_names=None, tag=None):
        super(WeightsHistHandler, self).__init__(model, tag=tag)
        self.layer_names = layer_names

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if self.layer_names is not None:
                if name.split('.')[0] not in self.layer_names:
                    continue

            name = name.replace(".", "/")
            logger.writer.add_histogram(
                tag="{}weights/{}".format(tag_prefix, name),
                values=p.data.detach().cpu().numpy(),
                global_step=global_step,
            )


class NumActivationsScalarHandler(BaseHandler):
    """Helper handler to log model's unit activation counts.
    Handler iterates over named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Args:
        model (torch.nn.Module): model to log weights
        tag (str, optional): common title for all produced plots. For example, 'generator'
    """

    def __init__(self, model, layer_names=None, tag=None):
        self.model = model
        self.tag = tag
        self.layer_names = layer_names
        self._num_iterations = defaultdict(int)
        self._num_activations_mean = defaultdict(int)

        # Register hooks
        self.hooks = {}
        for name, p in self.model.named_children():
            if self.layer_names is not None:
                if name not in self.layer_names:
                    continue

            self.hooks[name] = p.register_forward_hook(partial(self._hook_fn, layer_name=name))

    def _hook_fn(self, _, __, output, layer_name):
        self._num_iterations[layer_name] += 1
        self._num_activations_mean[layer_name] += (output.detach() > 0).sum(1).float().mean()
        # TODO
        # print('test', (output.detach() > 0).sum(1).float().mean())

    def reset(self, layer_name):
        self._num_iterations[layer_name] = 0
        self._num_activations_mean[layer_name] = 0

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'NumActivationsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for layer_name in self.layer_names:
            num_activations_mean = self._num_activations_mean[layer_name]
            num_iterations = self._num_iterations[layer_name]
            self.reset(layer_name)
            # TODO
            # print('test2', num_activations_mean / num_iterations)
            logger.writer.add_scalar(
                "{}num_activations_mean/{}".format(tag_prefix, layer_name),
                num_activations_mean / num_iterations,
                global_step
            )


class ActivationsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's unit activation counts.
    Handler iterates over named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, 'generator'
    """

    def __init__(self, model, reduction=torch.mean, layer_names=None, tag=None):
        super().__init__(model, reduction, tag=tag)
        self.layer_names = layer_names
        self.activations = {}

        # Register hooks
        self.hooks = {}
        for name, p in self.model.named_children():
            if self.layer_names is not None:
                if name not in self.layer_names:
                    continue

            self.hooks[name] = p.register_forward_hook(partial(self._hook_fn, layer_name=name))

    def _hook_fn(self, _, __, output, layer_name):
        self.activations[layer_name] = output

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'ActivationsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for layer_name, output in self.activations.items():
            activations = output.float()
            logger.writer.add_scalar(
                "{}activations_{}/{}".format(tag_prefix, self.reduction.__name__, layer_name),
                self.reduction(activations.detach()),
                global_step
            )


class ActivationsHistHandler(BaseWeightsHistHandler):
    """Helper handler to log model's activations as histograms.

    Args:
        model (torch.nn.Module): model to log activations
        tag (str, optional): common title for all produced plots. For example, 'generator'
    """

    def __init__(self, model, layer_names=None, tag=None):
        super().__init__(model, tag=tag)
        self.layer_names = layer_names
        self.activations = {}

        # Register hooks
        self.hooks = {}
        for name, p in self.model.named_children():
            if self.layer_names is not None:
                if name not in self.layer_names:
                    continue

            self.hooks[name] = p.register_forward_hook(partial(self._hook_fn, layer_name=name))

    def _hook_fn(self, _, __, output, layer_name):
        self.activations[layer_name] = output

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'ActivationsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for layer_name, output in self.activations.items():
            logger.writer.add_histogram(
                tag="{}activations/{}".format(tag_prefix, layer_name),
                values=output.detach().cpu().numpy(),
                global_step=global_step,
            )


class WeightsImageHandler(BaseHandler):
    def __init__(self, model: torch.nn.Module, input_shape):
        self.model = model
        self.input_shape = input_shape

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsImageHandler' works only with TensorboardLogger")

        with torch.no_grad():
            first_trainable = True
            for idx, (name, layer) in enumerate(self.model.named_children()):
                if type(layer) == torch.nn.Linear and first_trainable:
                    # Only visualize a Linear layer if it is the first layer after the input
                    weights = layer.weight.view(-1, *self.input_shape)
                    first_trainable = False
                elif type(layer) == torch.nn.Conv2d:
                    weights = layer.weight
                    if self.input_shape[0] > 1 and idx == 0:
                        weights = weights.view(-1, self.input_shape[0], *weights.shape[2:])
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

                global_step = engine.state.get_event_attrib_value(event_name)
                logger.writer.add_figure(tag=name + '/weight', figure=fig, global_step=global_step)
                plt.close()

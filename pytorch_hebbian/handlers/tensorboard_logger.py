from functools import partial

import torch
from ignite.contrib.handlers.base_logger import BaseWeightsScalarHandler, BaseWeightsHistHandler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

__all__ = [
    'WeightsScalarHandler',
    'WeightsHistHandler',
    'ActivationsScalarHandler',
    'ActivationsHistHandler',
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
            num_activated = (output > 0).sum(1).float()
            logger.writer.add_scalar(
                "{}num_activations_{}/{}".format(tag_prefix, self.reduction.__name__, layer_name),
                self.reduction(num_activated.detach()),
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

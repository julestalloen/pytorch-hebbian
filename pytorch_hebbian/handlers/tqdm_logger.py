import numbers
import warnings

import torch
from ignite.contrib.handlers.base_logger import BaseOutputHandler, BaseLogger


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics.

    Args:
        tag (str): common title for all produced plots. For example, 'training'
        metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, 'another_loss': loss2}` to label the plot
            with corresponding keys.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
    """

    def __init__(self, tag, metric_names="all", output_transform=None, global_step_transform=None):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, global_step_transform)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TqdmLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with TqdmLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        message = "{} epoch {}: ".format(self.tag.capitalize(), global_step)
        metrics_str = []
        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or isinstance(value, torch.Tensor) and value.ndimension() == 0:
                if value > 1e4:
                    metrics_str.append("{}={:.4e}".format(key, value))
                else:
                    metrics_str.append("{}={:.4f}".format(key, value))
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    metrics_str.append("{}{}={}".format(key, i, v.item()))
            else:
                warnings.warn(
                    "TqdmLogger output_handler can not log " "metrics value type {}".format(type(value))
                )
        logger.pbar.log_message(message + ", ".join(metrics_str))


class TqdmLogger(BaseLogger):
    """Tqdm logger to log messages using the progress bar."""

    def __init__(self, pbar):
        self.pbar = pbar

    def close(self):
        if self.pbar:
            self.pbar.close()
        self.pbar = None

    def _create_output_handler(self, *args, **kwargs):
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args, **kwargs):
        """Intentionally empty"""
        pass

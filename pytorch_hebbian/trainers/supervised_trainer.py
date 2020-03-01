from typing import Union, Optional

import torch
from ignite.engine import Events, create_supervised_trainer
from ignite.metrics import RunningAverage
from torch.optim.optimizer import Optimizer

from pytorch_hebbian import config
from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator
from pytorch_hebbian.trainers import Trainer
from pytorch_hebbian.visualizers import Visualizer


class SupervisedTrainer(Trainer):
    """Trains a model using classical supervised backpropagation."""

    def __init__(self, model: torch.nn.Module, optimizer: Optimizer, criterion, evaluator, lr_scheduler=None,
                 visualizer: Visualizer = None, device: Optional[Union[str, torch.device]] = None):
        device = self._get_device(device)
        engine = create_supervised_trainer(model, optimizer, criterion, device=device)

        self.train_evaluator = SupervisedEvaluator(model, criterion)
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

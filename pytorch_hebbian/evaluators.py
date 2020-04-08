import logging
from typing import Callable

import torch
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss

from pytorch_hebbian.trainers import SupervisedTrainer


class HebbianEvaluator:

    def __init__(self, model: torch.nn.Module, score_name: str, score_function: Callable,
                 init_function: Callable[[], tuple] = None, epochs: int = 100, supervised_from: int = None,
                 supervised_eval_every: int = 5):
        self.model = model
        self.score_name = score_name
        self.score_function = score_function
        if init_function is None:
            self.init_function = self._init_function
        else:
            self.init_function = init_function
        self.epochs = epochs
        self.supervised_from = supervised_from
        self.supervised_eval_every = supervised_eval_every

        self._init_metrics()

    @staticmethod
    def _init_function(model):
        """Default initialization function."""
        criterion = torch.nn.CrossEntropyLoss()
        evaluator = SupervisedEvaluator(model=model, criterion=criterion)
        train_evaluator = SupervisedEvaluator(model=model, criterion=criterion)
        optimizer = torch.optim.Adam(params=model.parameters())
        trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion,
                                    train_evaluator=train_evaluator, evaluator=evaluator)

        # Early stopping
        es_handler = EarlyStopping(patience=5,
                                   min_delta=0.0001,
                                   score_function=lambda engine: -engine.state.metrics['loss'],
                                   trainer=trainer.engine, cumulative_delta=True)
        evaluator.engine.add_event_handler(Events.COMPLETED, es_handler)

        return trainer, evaluator

    def _init_metrics(self):
        self.metrics = {}
        self.best_score = None

    def _init(self):
        self._trainer, self._evaluator = self.init_function(self.model)

        # Metric history saving
        @self._evaluator.engine.on(Events.COMPLETED)
        def save_best_metrics(engine):
            current_score = self.score_function(engine)
            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                self.metrics = engine.state.metrics
                logging.info("New best validation {} = {:.4f}.".format(self.score_name, self.best_score))

        self._init_metrics()

    def run(self, train_loader, val_loader, supervised_from):
        # Normally the trainer passes the supervised_from parameter to the evaluator. This value is overwritten if the
        #   parameter was manually passed on creation of the evaluator
        if self.supervised_from is not None:
            supervised_from = self.supervised_from
        logging.info(
            "Supervised training from layer '{}'.".format(list(self.model.named_children())[supervised_from][0]))

        self._init()

        layers = list(self.model.children())
        # Freeze the Hebbian trained layers
        for layer in layers[:supervised_from]:
            for param in layer.parameters():
                param.requires_grad = False

        # Re-initialize weights for the supervised layers
        for lyr in layers[supervised_from:]:
            try:
                lyr.reset_parameters()
            except AttributeError:
                pass

        self._trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=self.epochs,
                          eval_every=self.supervised_eval_every)


class SupervisedEvaluator:

    def __init__(self, model, criterion, metrics=None, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        if metrics is None:
            metrics = {
                'accuracy': Accuracy(),
                'loss': Loss(criterion)
            }

        self.engine = create_supervised_evaluator(model, metrics=metrics, device=device)
        self.metrics = {}  # engine.state.metrics only created on run

    def run(self, val_loader):
        self.engine.run(val_loader)
        self.metrics = self.engine.state.metrics

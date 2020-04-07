import logging
from typing import Callable

import torch
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss

from pytorch_hebbian.trainers import SupervisedTrainer


class HebbianEvaluator:

    def __init__(self, model: torch.nn.Module, init_func: Callable[[], tuple] = None, epochs=100, supervised_from=None,
                 supervised_eval_every=5):
        self.model = model
        if init_func is None:
            self.init_func = self._init_func
        else:
            self.init_func = init_func
        self.epochs = epochs
        self.supervised_from = supervised_from
        self.supervised_eval_every = supervised_eval_every

        self._init_metrics()

    @staticmethod
    def _init_func(model):
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
        self.metrics = {'loss': float('inf')}

    def _init(self):
        self._trainer, self._evaluator = self.init_func(self.model)

        # Metric history saving
        @self._evaluator.engine.on(Events.COMPLETED)
        def save_best_metrics(engine):
            loss = engine.state.metrics['loss']
            accuracy = engine.state.metrics['accuracy']

            if loss < self.metrics['loss']:
                self.metrics['loss'] = loss
                self.metrics['accuracy'] = accuracy
                logging.info("New best validation loss = {:.4f}.".format(loss))

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

    def run(self, val_loader):
        self.engine.run(val_loader)

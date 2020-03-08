import logging

import torch
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss

from pytorch_hebbian.trainers import SupervisedTrainer


class HebbianEvaluator:

    def __init__(self, model: torch.nn.Module, epochs=100, supervised_from=None, early_stopping_patience=5,
                 supervised_eval_every=5):
        self.model = model
        self.epochs = epochs
        self.supervised_from = supervised_from
        self.early_stopping_patience = early_stopping_patience
        self.supervised_eval_every = supervised_eval_every
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = SupervisedEvaluator(model=self.model, criterion=self.criterion)
        self.trainer = None
        self.metrics = None
        self._init_metrics()

    def _init_metrics(self):
        self.metrics = {'loss': float('inf')}

    def run(self, train_loader, val_loader, supervised_from):
        # Normally the trainer passes the supervised_from parameter to the evaluator. This value is overwritten if the
        #   parameter was manually passed on creation of the evaluator
        if self.supervised_from is not None:
            supervised_from = self.supervised_from
        logging.info(
            "Supervised training from layer '{}'.".format(list(self.model.named_children())[supervised_from][0]))

        # Init/reset metrics and engine state
        self._init_metrics()
        self.evaluator.engine.state = None

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

        # Create a new optimizer and trainer instance
        optimizer = torch.optim.Adam(params=self.model.parameters())
        self.trainer = SupervisedTrainer(model=self.model, optimizer=optimizer, criterion=self.criterion,
                                         evaluator=self.evaluator)

        # Metric history saving
        @self.evaluator.engine.on(Events.COMPLETED)
        def save_best_metrics(engine):
            loss = engine.state.metrics['loss']
            accuracy = engine.state.metrics['accuracy']

            if loss < self.metrics['loss']:
                self.metrics['loss'] = loss
                self.metrics['accuracy'] = accuracy
                logging.info("New best validation loss = {:.4f}.".format(loss))

        # Early stopping
        handler = EarlyStopping(patience=self.early_stopping_patience,
                                min_delta=0.0001,
                                score_function=lambda engine: -engine.state.metrics['loss'],
                                trainer=self.trainer.engine, cumulative_delta=True)
        self.evaluator.engine.add_event_handler(Events.COMPLETED, handler)

        self.trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=self.epochs,
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

import logging

import torch
from ignite.engine import Events
from ignite.handlers import EarlyStopping

from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator
from pytorch_hebbian.trainers import SupervisedTrainer


class HebbianEvaluator:

    def __init__(self, model, epochs=100, early_stopping_patience=5, supervised_eval_every=5):
        self.model = model
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.supervised_eval_every = supervised_eval_every
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = SupervisedEvaluator(model=self.model, criterion=self.criterion)
        self.trainer = None
        self.metrics = None
        self._init_metrics()

    def _init_metrics(self):
        self.metrics = {'loss': float('inf')}

    def run(self, train_loader, val_loader):
        # Init/reset metrics and engine state
        self._init_metrics()
        self.evaluator.engine.state = None

        layers = list(self.model.children())
        # Freeze all but final layer of the model
        for layer in layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        # Re-initialize weights for final layer
        layers[-1].reset_parameters()

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
                                score_function=lambda engine: -engine.state.metrics['loss'],
                                trainer=self.trainer.engine, cumulative_delta=True)
        self.evaluator.engine.add_event_handler(Events.COMPLETED, handler)

        self.trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=self.epochs,
                         eval_every=self.supervised_eval_every)

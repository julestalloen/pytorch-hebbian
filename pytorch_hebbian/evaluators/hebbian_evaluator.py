import logging

import torch
from ignite.engine import Events
from ignite.handlers import EarlyStopping

from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator
from pytorch_hebbian.trainers import SupervisedTrainer


class HebbianEvaluator:

    def __init__(self, model, epochs=100):
        self.model = model
        self.epochs = epochs
        self.metrics = {'loss': float('inf')}

    def run(self, train_loader, val_loader):
        # Freeze all but final layer
        for layer in list(self.model.children())[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters())
        evaluator = SupervisedEvaluator(model=self.model, criterion=criterion)
        trainer = SupervisedTrainer(model=self.model, optimizer=optimizer, criterion=criterion, evaluator=evaluator)

        # Metric history saving
        @evaluator.engine.on(Events.COMPLETED)
        def save_best_metrics(engine):
            loss = engine.state.metrics['loss']
            accuracy = engine.state.metrics['accuracy']

            if loss < self.metrics['loss']:
                self.metrics['loss'] = loss
                self.metrics['accuracy'] = accuracy
                logging.info("New best loss: {:.4f}.".format(loss))

        # Early stopping
        handler = EarlyStopping(patience=4, score_function=lambda engine: -engine.state.metrics['loss'],
                                trainer=trainer.engine, cumulative_delta=True)
        evaluator.engine.add_event_handler(Events.COMPLETED, handler)

        trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=self.epochs, eval_every=5)

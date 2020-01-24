import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

import config


class SupervisedTrainer:

    def __init__(self, model, optimizer, criterion, train_loader, val_loader, visualizer, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.visualizer = visualizer

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
        # TODO: replace with custom supervised_evaluator class
        self.evaluator = create_supervised_evaluator(model, metrics={
            'accuracy': Accuracy(),
            'loss': Loss(criterion)
        }, device=device)

        RunningAverage(output_transform=lambda x: x).attach(self.trainer, 'loss')
        self.pbar = ProgressBar(persist=True, bar_format=config.IGNITE_BAR_FORMAT)
        self.pbar.attach(self.trainer, metric_names='all')

        self._register_handlers()

    def _register_handlers(self):
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.evaluator.run(self.train_loader)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']

            self.pbar.log_message(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss)
            )
            self.visualizer.writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
            self.visualizer.writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.evaluator.run(self.val_loader)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']

            self.pbar.log_message(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss)
            )
            self.pbar.n = self.pbar.last_print_n = 0
            self.visualizer.writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
            self.visualizer.writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

    def run(self, epochs):
        self.trainer.run(self.train_loader, max_epochs=epochs)

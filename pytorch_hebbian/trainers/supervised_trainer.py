import logging

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer
from ignite.metrics import RunningAverage

import config
from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator


class SupervisedTrainer:

    def __init__(self, model, optimizer, criterion, evaluator, visualizer=None, device=None):
        self.evaluator = evaluator
        self.train_evaluator = SupervisedEvaluator(model, criterion)
        self.visualizer = visualizer
        self.train_loader = None
        self.val_loader = None
        self.eval_every = None

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        logging.info("Device set to '{}'.".format(device))

        self.engine = create_supervised_trainer(model, optimizer, criterion, device=device)

        RunningAverage(output_transform=lambda x: x).attach(self.engine, 'loss')
        self.pbar = ProgressBar(persist=True, bar_format=config.IGNITE_BAR_FORMAT)
        self.pbar.attach(self.engine, metric_names='all')

        self._register_handlers()

    def _register_handlers(self):
        @self.engine.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            if engine.state.epoch % self.eval_every == 0:
                self.train_evaluator.run(self.train_loader)
                metrics = self.train_evaluator.engine.state.metrics
                avg_accuracy = metrics['accuracy']
                avg_loss = metrics['loss']

                self.pbar.log_message(config.TRAIN_REPORT_FORMAT.format(engine.state.epoch, avg_accuracy, avg_loss))
                if self.visualizer is not None:
                    self.visualizer.writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
                    self.visualizer.writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

        @self.engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            if engine.state.epoch % self.eval_every == 0:
                self.evaluator.run(self.val_loader)
                metrics = self.evaluator.engine.state.metrics
                avg_accuracy = metrics['accuracy']
                avg_loss = metrics['loss']

                self.pbar.log_message(config.EVAL_REPORT_FORMAT.format(engine.state.epoch, avg_accuracy, avg_loss))
                if self.visualizer is not None:
                    self.visualizer.writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
                    self.visualizer.writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

                self.pbar.n = self.pbar.last_print_n = 0

    def run(self, train_loader, val_loader, epochs, eval_every=1):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_every = eval_every
        logging.info('Evaluating every {} epoch(s).'.format(self.eval_every))
        self.engine.run(train_loader, max_epochs=epochs)

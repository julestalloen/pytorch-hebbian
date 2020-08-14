import logging
from abc import ABC
from typing import Callable

import torch
from ignite.engine import Events, Engine, State, create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss

from pytorch_hebbian import utils
from pytorch_hebbian.trainers import SupervisedTrainer


class SimpleEngine(Engine):
    """Custom engine with custom run function.

    This engine has only metrics in its state and only fires 2 events.
    """

    def __init__(self, run_function: Callable):
        super().__init__(process_function=lambda x, y: None)
        self._allowed_events = [Events.STARTED, Events.COMPLETED]
        self._run_function = run_function

    def run(self, *args, **kwargs):
        self._fire_event(Events.STARTED)
        self._run_function(*args, **kwargs)
        self._fire_event(Events.COMPLETED)


class Evaluator(ABC):
    def __init__(self):
        self.engine = None
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def attach(self, engine, event_name, *args, **kwargs):
        if event_name not in State.event_to_attr:
            raise RuntimeError("Unknown event name '{}'".format(event_name))

        return engine.add_event_handler(event_name, self, *args, **kwargs)

    def run(self, *args, **kwargs):
        self.engine.run(*args, **kwargs)

    def __call__(self, engine, *args, **kwargs):
        self.run(*args, **kwargs)


class HebbianEvaluator(Evaluator):
    def __init__(self, model: torch.nn.Module, score_name: str, score_function: Callable,
                 init_function: Callable[[torch.nn.Module], tuple] = None, epochs: int = 100,
                 supervised_from: int = None):
        super().__init__()
        self.model = model
        self.score_name = score_name
        self.score_function = score_function
        if init_function is None:
            self.init_function = self._init_function
        else:
            self.init_function = init_function
        self.epochs = epochs
        self.supervised_from = supervised_from

        self.engine = self.create_hebbian_evaluator(self._run)
        self._init_metrics()

    @staticmethod
    def create_hebbian_evaluator(run_function) -> Engine:
        return SimpleEngine(run_function=run_function)

    @staticmethod
    def _init_function(model):
        """Default initialization function."""
        criterion = torch.nn.CrossEntropyLoss()
        evaluator = SupervisedEvaluator(model=model, criterion=criterion)
        train_evaluator = SupervisedEvaluator(model=model, criterion=criterion)
        optimizer = torch.optim.Adam(params=model.parameters())
        trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion)

        # Early stopping
        es_handler = EarlyStopping(patience=5,
                                   min_delta=0.0001,
                                   score_function=lambda engine: -engine.state.metrics['loss'],
                                   trainer=trainer.engine, cumulative_delta=True)
        evaluator.engine.add_event_handler(Events.COMPLETED, es_handler)

        return trainer, train_evaluator, evaluator

    def _init_metrics(self):
        # self.metrics = {}
        self.best_score = None

    def _init(self, train_loader, val_loader):
        self._trainer, self._train_evaluator, self._evaluator = self.init_function(self.model)

        # Attach evaluators
        self._train_evaluator.attach(self._trainer.engine, Events.EPOCH_COMPLETED, train_loader)
        self._evaluator.attach(self._trainer.engine, Events.EPOCH_COMPLETED, val_loader)

        # Metric history saving
        @self._evaluator.engine.on(Events.COMPLETED)
        def save_best_metrics(eval_engine):
            current_score = self.score_function(eval_engine)
            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                self.engine.state.metrics = eval_engine.state.metrics
                self.logger.info("New best validation {} = {:.4f}.".format(self.score_name, self.best_score))

        self._init_metrics()

    def _run(self, train_loader, val_loader):
        self.logger.info(
            "Supervised training from layer '{}'.".format(list(self.model.named_children())[self.supervised_from][0]))

        self._init(train_loader, val_loader)

        layers = list(self.model.children())
        # Freeze the Hebbian trained layers
        for layer in layers[:self.supervised_from]:
            for param in layer.parameters():
                param.requires_grad = False

        # Re-initialize weights for the supervised layers
        for lyr in layers[self.supervised_from:]:
            try:
                lyr.reset_parameters()
            except AttributeError:
                pass

        self._trainer.run(train_loader=train_loader, epochs=self.epochs)


class SupervisedEvaluator(Evaluator):
    def __init__(self, model, criterion, metrics=None, device=None):
        super().__init__()
        self.device = utils.get_device(device)

        if metrics is None:
            metrics = {
                'accuracy': Accuracy(),
                'loss': Loss(criterion)
            }

        self.engine = create_supervised_evaluator(model, metrics=metrics, device=self.device)
        # self.metrics = self.engine.state.metrics

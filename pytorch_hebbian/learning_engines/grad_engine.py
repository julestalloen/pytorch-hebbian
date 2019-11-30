import logging

from torch.nn import Module
from torch.utils.data import DataLoader

from pytorch_hebbian.learning_engines.learning_engine import LearningEngine


class HebbianEngine(LearningEngine):

    def __init__(self, optimizer, lr_scheduler, criterion, evaluator):
        super().__init__(optimizer, lr_scheduler, evaluator)
        self.criterion = criterion

    def train(self, model: Module, data_loader: DataLoader, epochs: int,
              eval_every: int = None, checkpoint_every: int = None):
        pass

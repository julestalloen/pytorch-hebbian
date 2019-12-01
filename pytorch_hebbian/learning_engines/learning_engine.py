from abc import ABC, abstractmethod
import logging

import torch

import config


class LearningEngine(ABC):

    def __init__(self, optimizer, lr_scheduler, evaluator=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator

    @abstractmethod
    def train(self, *args):
        pass

    def eval(self):
        if self.evaluator is None:
            logging.warning('No evaluator specified.')
        else:
            logging.info('Evaluating...')
            self.evaluator.model.eval()
            self.evaluator.run()

    @staticmethod
    def checkpoint(model):
        logging.info('Saving checkpoint...')
        torch.save(model.state_dict(), config.MODELS_DIR)

from abc import ABC, abstractmethod
import logging


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
            logging.info('No evaluator specified.')
        else:
            logging.info('Evaluating...')
            self.evaluator.model.eval()
            self.evaluator.run()

    @staticmethod
    def checkpoint(model):
        logging.info('Saving checkpoint...')
        # TODO

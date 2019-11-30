from abc import ABC, abstractmethod
import logging


class LearningEngine(ABC):

    def __init__(self, optimizer, lr_scheduler, evaluator):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator

    @abstractmethod
    def train(self, *args):
        pass

    def eval(self):
        logging.info('Evaluating...')
        if self.evaluator is None:
            logging.warning('No evaluator specified!')
        else:
            self.evaluator.run()

    @staticmethod
    def checkpoint(model):
        logging.info('Saving checkpoint...')
        # TODO

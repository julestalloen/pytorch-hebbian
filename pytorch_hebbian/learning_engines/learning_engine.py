from abc import ABC, abstractmethod
import logging


class LearningEngine(ABC):

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def validate(self, *args):
        pass

    @staticmethod
    def checkpoint(model):
        logging.info('Saving checkpoint...')
        # TODO

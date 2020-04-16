import logging
from abc import ABC, abstractmethod


class LearningRule(ABC):

    def __init__(self):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def init_layers(self, model):
        pass

    @abstractmethod
    def update(self, x, w):
        pass

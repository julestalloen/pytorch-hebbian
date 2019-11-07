from abc import ABC, abstractmethod


class LearningRule(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def update(self, x, w):
        pass

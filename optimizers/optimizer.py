from abc import ABC, abstractmethod
import collections


class Optimizer(ABC):

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.updates = collections.deque(maxlen=10)

    @abstractmethod
    def get_updates(self, d_w, epoch, epochs):
        pass

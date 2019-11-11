from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.updates = []

    @abstractmethod
    def get_updates(self, d_w, epoch, epochs):
        pass

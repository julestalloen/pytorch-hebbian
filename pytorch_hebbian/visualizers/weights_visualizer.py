from abc import ABC, abstractmethod


class WeightsVisualizer(ABC):

    @abstractmethod
    def update(self, weights, shape, step):
        pass

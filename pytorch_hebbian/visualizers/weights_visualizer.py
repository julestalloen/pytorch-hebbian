from abc import ABC, abstractmethod


class WeightsVisualizer(ABC):

    @abstractmethod
    def visualize_weights(self, weights, shape, step):
        pass

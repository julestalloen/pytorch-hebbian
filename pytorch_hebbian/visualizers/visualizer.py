from abc import ABC


class Visualizer(ABC):
    """Abstract base visualizer class to be passed to a trainer."""

    def visualize_metrics(self, metrics, epoch: int):
        pass

    def visualize_weights(self, layers: zip, input_shape, step: int):
        pass

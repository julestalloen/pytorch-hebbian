from abc import ABC, abstractmethod


class Visualizer(ABC):

    @abstractmethod
    def update(self, *args):
        pass

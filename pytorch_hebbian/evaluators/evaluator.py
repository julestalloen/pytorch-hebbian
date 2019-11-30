from abc import ABC, abstractmethod


class Evaluator(ABC):

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    @abstractmethod
    def run(self):
        pass

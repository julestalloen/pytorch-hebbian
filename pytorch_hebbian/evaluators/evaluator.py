from abc import ABC, abstractmethod
import logging

import torch


class Evaluator(ABC):

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logging.info('Device set to {}.'.format(self.device))

    @abstractmethod
    def run(self) -> dict:
        pass

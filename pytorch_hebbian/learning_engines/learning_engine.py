from abc import ABC, abstractmethod
import logging
import os
import time

import torch

import config


class LearningEngine(ABC):

    def __init__(self, optimizer, lr_scheduler, evaluator=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def train(self, *args):
        pass

    def eval(self):
        if self.evaluator is None:
            logging.warning('No evaluator specified.')
            return None
        else:
            logging.info('Evaluating...')
            self.evaluator.model.eval()
            results = self.evaluator.run()

            print('\n' + ', '.join(['{} = {}'.format(key, value) for key, value in results.items()]))
            return results

    @staticmethod
    def checkpoint(model, stats: dict = None):
        logging.info('Saving checkpoint...')
        if not os.path.exists(config.MODELS_DIR):
            os.makedirs(config.MODELS_DIR)

        model_name = 'm'
        if stats is not None:
            for k, v in stats.items():
                model_name += '-{}{:.2f}'.format(k, v)

        model_name += '-' + time.strftime("%Y%m%d-%H%M%S")
        torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, model_name + '.pth'))

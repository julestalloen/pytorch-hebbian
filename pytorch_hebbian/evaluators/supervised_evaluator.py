import torch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


class SupervisedEvaluator:

    def __init__(self, model, criterion, metrics=None, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        if metrics is None:
            metrics = {
                'accuracy': Accuracy(),
                'loss': Loss(criterion)
            }

        self.engine = create_supervised_evaluator(model, metrics=metrics, device=device)

    def run(self, val_loader):
        self.engine.run(val_loader)

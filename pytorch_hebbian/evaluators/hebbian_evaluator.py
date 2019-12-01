import torch

from pytorch_hebbian.learning_engines.supervised_engine import SupervisedEngine
from pytorch_hebbian.evaluators.evaluator import Evaluator
from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator


class HebbianEvaluator(Evaluator):

    def __init__(self, model, data_loader):
        super().__init__(model, data_loader)
        optimizer = torch.optim.Adam(model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        evaluator = SupervisedEvaluator(data_loader=data_loader, model=model)
        self.supervised_engine = SupervisedEngine(optimizer=optimizer,
                                                  lr_scheduler=lr_scheduler,
                                                  criterion=criterion,
                                                  evaluator=evaluator)

    def run(self):
        # Freeze all but final layer
        for layer in list(self.model.children())[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        # Train with gradient descent
        # TODO

        # Evaluate
        # TODO

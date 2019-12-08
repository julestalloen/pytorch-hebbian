import torch

from pytorch_hebbian.learning_engines.supervised_engine import SupervisedEngine
from pytorch_hebbian.evaluators.evaluator import Evaluator
from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator


class HebbianEvaluator(Evaluator):

    def __init__(self, model, data_loader, epochs=100):
        super().__init__(model, data_loader)
        optimizer = torch.optim.Adam(model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = SupervisedEvaluator(data_loader=data_loader, model=model, loss_criterion=criterion)
        self.supervised_engine = SupervisedEngine(optimizer=optimizer,
                                                  lr_scheduler=lr_scheduler,
                                                  criterion=criterion,
                                                  evaluator=self.evaluator)
        self.epochs = epochs

    def run(self):
        # Freeze all but final layer
        for layer in list(self.model.children())[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        # TODO: idea, reset weights for each run
        #   https://discuss.pytorch.org/t/reset-model-weights/19180/2

        # Train with gradient descent and evaluate
        self.supervised_engine.train(model=self.model, data_loader=self.data_loader, epochs=self.epochs, eval_every=10)

        return {'loss': min(self.evaluator.losses), 'acc': max(self.evaluator.accuracies)}

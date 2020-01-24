import torch

from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator
from pytorch_hebbian.trainers import SupervisedTrainer


class HebbianEvaluator:

    def __init__(self, model, epochs=100):
        self.model = model
        self.epochs = epochs
        self.metrics = {}

    def run(self, train_loader, val_loader):
        # Freeze all but final layer
        for layer in list(self.model.children())[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters())
        evaluator = SupervisedEvaluator(model=self.model, criterion=criterion)
        trainer = SupervisedTrainer(model=self.model, optimizer=optimizer, criterion=criterion, evaluator=evaluator)

        trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=self.epochs, eval_every=10)

        self.metrics = evaluator.engine.state.metrics

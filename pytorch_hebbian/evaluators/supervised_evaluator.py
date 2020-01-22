import torch
from tqdm import tqdm

from pytorch_hebbian.evaluators.evaluator import Evaluator


class SupervisedEvaluator(Evaluator):

    def __init__(self, model, data_loader, loss_criterion):
        super().__init__(model, data_loader)
        self.loss_criterion = loss_criterion
        self.accuracies = []
        self.losses = []

    def run(self):
        running_loss = 0.0
        running_corrects = 0

        progress_bar = tqdm(self.data_loader, desc='Evaluating')
        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device).view(inputs.size(0), -1)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)

                # Statistics
                loss = self.loss_criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

        self.losses.append(running_loss / len(self.data_loader.dataset))
        self.accuracies.append(running_corrects.double() / len(self.data_loader.dataset))

        return {'loss': self.losses[-1], 'acc': self.accuracies[-1]}

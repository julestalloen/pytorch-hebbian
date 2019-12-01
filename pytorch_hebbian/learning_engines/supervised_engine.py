import logging

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_hebbian.learning_engines.learning_engine import LearningEngine


class SupervisedEngine(LearningEngine):

    def __init__(self, optimizer, lr_scheduler, criterion, evaluator=None):
        super().__init__(optimizer, lr_scheduler, evaluator)
        self.criterion = criterion

    def train(self, model: Module, data_loader: DataLoader, epochs: int,
              eval_every: int = None, checkpoint_every: int = None):
        model.train()

        # Training loop
        for epoch in range(epochs):
            vis_epoch = epoch + 1
            running_loss = 0.0

            logging.info("Learning rate(s) = {}.".format(self.lr_scheduler.get_lr()))

            progress_bar = tqdm(data_loader, desc='Epoch {}/{}'.format(vis_epoch, epochs))
            for i, data in enumerate(progress_bar):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Calculate the loss
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Back propagation and optimize
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            self.lr_scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            logging.info('Train loss: {:.4f}'.format(epoch_loss))

            # TODO save or return best model

            # Evaluation
            if eval_every is not None:
                if vis_epoch % eval_every == 0:
                    stats = self.eval()
                    print(stats)

            # Checkpoint saving
            if checkpoint_every is not None:
                if vis_epoch % checkpoint_every == 0:
                    self.checkpoint(model)

        return model

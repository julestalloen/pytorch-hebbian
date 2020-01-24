import logging

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events

import config
from pytorch_hebbian.utils import data


class HebbianTrainer:

    def __init__(self, model, learning_rule, optimizer, lr_scheduler, evaluator=None, visualizer=None, device=None):
        self.evaluator = evaluator
        self.visualizer = visualizer
        self.train_loader = None
        self.val_loader = None
        # TODO: use learning rate scheduler

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        logging.info("Device set to '{}'.".format(device))

        # TODO: support multiple layers
        self.layer = None
        for layer in list(model.children())[:-1]:
            if type(layer) == torch.nn.Linear:
                self.layer = layer
                weights = layer.weight
                weights.data.normal_(mean=0.0, std=1.0)
                weights_np = weights.detach().numpy()
                logging.info("Updating layer '{}' with shape {}.".format(layer, weights_np.shape))

        self.engine = self.create_hebbian_trainer(model, learning_rule, optimizer, device)

        self.pbar = ProgressBar(persist=True, bar_format=config.IGNITE_BAR_FORMAT)
        self.pbar.attach(self.engine)

        self._register_handlers()

    def create_hebbian_trainer(self, model, learning_rule, optimizer, device=None, non_blocking=False,
                               prepare_batch=data.prepare_batch,
                               output_transform=lambda x, y, y_pred: 0):
        def _update(_, batch):
            model.train()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)

            inputs = np.reshape(x.squeeze(), (x.shape[0], -1))
            weights_np = self.layer.weight.detach().numpy()
            d_p = torch.from_numpy(learning_rule.update(inputs, weights_np))
            optimizer.local_step(d_p)

            return output_transform(x, y, y_pred)

        return Engine(_update)

    def _register_handlers(self):
        @self.engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            pass
            # TODO

    def run(self, train_loader, val_loader, epochs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.engine.run(train_loader, max_epochs=epochs)

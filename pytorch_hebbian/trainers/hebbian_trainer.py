import logging

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events

import config
from pytorch_hebbian.utils import data


class HebbianTrainer:

    def __init__(self, model, learning_rule, optimizer, lr_scheduler, evaluator=None, visualizer=None, device=None):
        self.evaluator = evaluator
        self.visualizer = visualizer
        self.train_loader = None
        self.val_loader = None
        self.eval_every = None
        self.vis_weights_every = None
        self.input_shape = None

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
                logging.info("Updating layer '{}' with shape {}.".format(layer, weights.shape))

        self.engine = self.create_hebbian_trainer(model, learning_rule, optimizer, device)

        self.scheduler = LRScheduler(lr_scheduler)
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self.scheduler)

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

            inputs = torch.reshape(x, (x.shape[0], -1))
            d_p = learning_rule.update(inputs, self.layer.weight)
            optimizer.local_step(d_p)

            return output_transform(x, y, y_pred)

        return Engine(_update)

    def _register_handlers(self):
        @self.engine.on(Events.EPOCH_STARTED)
        def log_learning_rate(engine):
            logging.debug('Learning rate: {}.'.format(round(self.scheduler.get_param(), 6)))
            self.visualizer.writer.add_scalar('learning_rate', self.scheduler.get_param(), engine.state.epoch - 1)

        @self.engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            if engine.state.epoch % self.eval_every == 0:
                self.evaluator.run(self.train_loader, self.val_loader)
                metrics = self.evaluator.metrics
                avg_accuracy = metrics['accuracy']
                avg_loss = metrics['loss']

                self.pbar.log_message(config.EVAL_REPORT_FORMAT.format(engine.state.epoch, avg_accuracy, avg_loss))
                self.visualizer.writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
                self.visualizer.writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

                self.pbar.n = self.pbar.last_print_n = 0

        @self.engine.on(Events.ITERATION_COMPLETED)
        def visualize_weights(engine):
            if engine.state.iteration % self.vis_weights_every == 0:
                self.visualizer.visualize_weights(self.layer.weight, self.input_shape, engine.state.epoch)

    def run(self, train_loader, val_loader, epochs, eval_every=1, vis_weights_every=20):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_every = eval_every
        self.vis_weights_every = vis_weights_every
        self.input_shape = tuple(next(iter(self.train_loader))[0].shape[1:])
        logging.info('Received {} training and {} validation samples.'.format(len(train_loader.dataset),
                                                                              len(val_loader.dataset)))
        logging.info('Training {} epochs, evaluating every {} epoch(s).'.format(epochs, self.eval_every))
        logging.debug('Visualizing weights every {} epoch(s).'.format(self.vis_weights_every))
        self.engine.run(train_loader, max_epochs=epochs)

import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace

# from scipy import signal
import torch
from ignite.contrib.handlers import LRScheduler
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping
# import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import HebbianEvaluator, SupervisedEvaluator
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.metrics import UnitConvergence
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer, SupervisedTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def main(args: Namespace, params: dict):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'heb-{}'.format(identifier)
    logging.info("Starting run '{}'.".format(run))

    # Loading the model and possibly initial weights
    model = models.dense_net1
    if args.initial_weights is not None:
        model = utils.load_weights(model, os.path.join(PATH, args.initial_weights))
        freeze_layers = ['1']
    else:
        freeze_layers = None

    # Device selection
    device = utils.get_device(args.device)
    logging.info("Device set to '{}'.".format(device))
    model.to(device)

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = Subset(dataset, [i for i in range(10000)])
    train_dataset, val_dataset = utils.split_dataset(dataset, val_split=params['val_split'])
    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # Analyze dataset
    data_batch = next(iter(train_loader))[0]
    logging.debug("Data batch min: {:.4f}, max: {:.4f}.".format(torch.min(data_batch),
                                                                torch.max(data_batch)))
    logging.debug("Data batch mean: {1:.4f}, std: {0:.4f}.".format(*torch.std_mean(data_batch)))

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run, log_dir=args.log_dir)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the learning rule, optimizer, learning rate scheduler, evaluator and trainer
    epochs = params['epochs']
    # learning_rule = {
    #     '0': KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm']),
    #     '3': KrotovsRule(delta=0.2, k=params['k'], norm=params['norm']),
    # }
    learning_rule = KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm'])
    optimizer = Local(named_params=model.named_parameters(), lr=params['lr'])
    # TODO: Typing issue should be fixed in future pytorch update
    #   https://github.com/pytorch/pytorch/issues/32645
    # noinspection PyTypeChecker
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs)
    lr_scheduler = LRScheduler(lr_scheduler)

    # Initialization function called before each evaluation run of the Hebbian evaluator
    def init_function(h_model):
        h_criterion = torch.nn.CrossEntropyLoss()
        h_evaluator = SupervisedEvaluator(model=h_model, criterion=h_criterion, device=device)
        h_train_evaluator = SupervisedEvaluator(model=h_model, criterion=h_criterion, device=device)
        h_optimizer = torch.optim.Adam(params=h_model.parameters(), lr=1e-4)
        h_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(h_optimizer, 'max', verbose=True, patience=4,
                                                                    factor=0.2)
        h_trainer = SupervisedTrainer(model=h_model, optimizer=h_optimizer, criterion=h_criterion,
                                      train_evaluator=h_train_evaluator, evaluator=h_evaluator, device=device)

        # Learning rate scheduling
        # The PyTorch Ignite LRScheduler class does not work with ReduceLROnPlateau
        h_evaluator.engine.add_event_handler(Events.COMPLETED,
                                             lambda engine: h_lr_scheduler.step(engine.state.metrics['accuracy']))

        # Model checkpoints
        h_handler = ModelCheckpoint(config.MODELS_DIR, run, n_saved=1, create_dir=True, require_empty=False,
                                    score_name='acc', score_function=lambda engine: engine.state.metrics['accuracy'],
                                    global_step_transform=global_step_from_engine(trainer.engine))
        h_evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, h_handler, {'m': model})

        # Early stopping
        h_es_handler = EarlyStopping(patience=10,
                                     min_delta=0.0001,
                                     score_function=lambda engine: engine.state.metrics['accuracy'],
                                     trainer=h_trainer.engine, cumulative_delta=True)
        h_es_handler.logger.setLevel(logging.INFO)
        h_evaluator.engine.add_event_handler(Events.COMPLETED, h_es_handler)

        # Trainer progress bar persistence
        # if args.no_persist_pb:
        h_trainer.pbar.persist = False

        return h_trainer, h_evaluator

    evaluator = HebbianEvaluator(model=model, score_name='accuracy',
                                 score_function=lambda engine: engine.state.metrics['accuracy'], epochs=500,
                                 init_function=init_function)
    trainer = HebbianTrainer(model=model, learning_rule=learning_rule, optimizer=optimizer, supervised_from=-1,
                             freeze_layers=freeze_layers, evaluator=evaluator, visualizer=visualizer,
                             device=device)

    # Metrics
    UnitConvergence(model[1], learning_rule.norm).attach(trainer.engine, 'unit_conv')
    if args.gpu_metrics and device == 'cuda':
        GpuInfo().attach(trainer.engine, name='gpu')

    # Trainer progress bar persistence
    if args.no_persist_pb:
        trainer.pbar.persist = False

    # Adding handlers for learning rate scheduling, model checkpoints and visualizing
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    mc_handler = ModelCheckpoint(config.MODELS_DIR, run, n_saved=1, create_dir=True, require_empty=False,
                                 global_step_transform=global_step_from_engine(trainer.engine))
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, mc_handler, {'m': model})

    @trainer.engine.on(Events.EPOCH_STARTED)
    def log_learning_rate(engine):
        visualizer.writer.add_scalar('learning_rate', lr_scheduler.get_param(), engine.state.epoch - 1)

    @trainer.engine.on(Events.EPOCH_COMPLETED)
    def log_unit_convergence(engine):
        weights = model[1].weight.detach()
        sums = torch.sum(torch.pow(torch.abs(weights), params['norm']), 1).cpu()
        visualizer.writer.add_scalar('unit_convergence', engine.state.metrics['unit_conv'], engine.state.epoch)

        fig = plt.figure()
        plt.bar(range(sums.shape[0]), sums)
        plt.xlabel("{:.2f}% of hidden units 'converged'".format(engine.state.metrics['unit_conv'] * 100))
        plt.ylabel("Sum of incoming weights")
        fig.tight_layout()
        image = utils.plot_to_img(fig)
        visualizer.writer.add_image('unit_weight_sum', image, engine.state.epoch)

    # @trainer.engine.on(Events.STARTED)
    # @trainer.engine.on(Events.EPOCH_COMPLETED)
    # def log_unit_stats(engine):
    #     # TODO: WIP
    #     # https://en.wikipedia.org/wiki/Digital_image_correlation_and_tracking
    #     layer = model[1]
    #     weights = layer.weight.detach()
    #     weights = weights.view(weights.shape[0], 28, 28)
    #     # kernels = weights.view(weights.shape[0], -1)
    #     kernels = torch.squeeze(weights)
    #     num_kernels = kernels.shape[0]
    #     # kernel_size = kernels.shape[1]
    #     corr = np.zeros((num_kernels, num_kernels))
    #
    #     # Zero-normalized cross-correlation (ZNCC)
    #     for i in range(num_kernels):
    #         for j in range(num_kernels):
    #             # sim[i, j] = torch.sum(np.abs(kernels[i] - kernels[j])) / kernel_size
    #             kernel1 = kernels[i].numpy()
    #             kernel2 = kernels[j].numpy()
    #             kernel1 = (kernel1 - np.mean(kernel1)) / np.std(kernel1)
    #             kernel2 = (kernel2 - np.mean(kernel2)) / np.std(kernel2)
    #             corr_mat = signal.correlate2d(kernel1, kernel2)
    #             corr[i, j] = np.amax(corr_mat) / 784
    #
    #     fig = plt.figure()
    #     mat = plt.imshow(corr, cmap='Reds', vmin=0)
    #     plt.colorbar(mat)
    #     fig.tight_layout()
    #     image = utils.plot_to_img(fig)
    #     visualizer.writer.add_image('kernel_correlation', image, engine.state.epoch)

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=epochs, eval_every=500)

    # Save the final parameters with its corresponding metrics
    visualizer.writer.add_hparams(params, evaluator.engine.state.metrics)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, required=True,
                        help='use a preset json file to specify parameters')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enable debug logging')
    parser.add_argument("--log_dir", type=str, default=config.TENSORBOARD_DIR,
                        help="log directory for Tensorboard log output")
    parser.add_argument('--initial_weights', type=str,
                        help='model weights to initialize training')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help="'cuda' (GPU) or 'cpu'")
    parser.add_argument('--gpu_metrics', action='store_true', default=False,
                        help='enable gpu metrics in the trainer progress bar')
    parser.add_argument('--no_persist_pb', action='store_true', default=False,
                        help="don't persist the trainer progress bar")
    args_ = parser.parse_args()

    with open(os.path.join(PATH, args_.json)) as f:
        params_ = json.load(f)['params']

    logging.basicConfig(level=logging.DEBUG if args_.debug else logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    logging.debug("Arguments: {}.".format(vars(args_)))
    logging.debug("Parameters: {}.".format(params_))

    main(args_, params_)

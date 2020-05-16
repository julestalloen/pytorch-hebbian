import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace

# from scipy import signal
import torch
from ignite.contrib.handlers import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OptimizerParamsHandler
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping
# import numpy as np
from matplotlib import pyplot as plt

import data
import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import HebbianEvaluator, SupervisedEvaluator
from pytorch_hebbian.handlers.tensorboard_logger import *
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.metrics import UnitConvergence
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer, SupervisedTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def attach_handlers(run, model, optimizer, lr_scheduler, trainer, _, visualizer, params):
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    mc_handler = ModelCheckpoint(config.MODELS_DIR, run, n_saved=1, create_dir=True, require_empty=False,
                                 global_step_transform=global_step_from_engine(trainer.engine))
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, mc_handler, {'m': model})

    @trainer.engine.on(Events.EPOCH_COMPLETED)
    def log_unit_convergence(engine):
        weights = model[1].weight.detach()
        sums = torch.sum(torch.pow(torch.abs(weights), params['norm']), 1).cpu()
        visualizer.writer.add_scalar('unit_convergence', engine.state.metrics['unit_conv'], engine.state.epoch)

        fig = plt.figure()
        plt.bar(range(sums.shape[0]), sums)
        plt.xlabel("hidden units")
        plt.ylabel("Sum of incoming weights")
        fig.tight_layout()
        visualizer.writer.add_figure('unit_weight_sum', fig, engine.state.epoch)

    # Create a TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=os.path.join(config.TENSORBOARD_DIR, run))
    tb_logger.writer = visualizer.writer
    tb_logger.attach(trainer.engine, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.EPOCH_STARTED)
    tb_logger.attach(trainer.engine,
                     log_handler=WeightsScalarHandler(model, layer_names=['linear1', 'linear2']),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer.engine,
                     log_handler=WeightsHistHandler(model, layer_names=['linear1', 'linear2']),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer.engine,
                     log_handler=ActivationsHistHandler(model, layer_names=['batch_norm', 'repu']),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer.engine,
                     log_handler=NumActivationsScalarHandler(model, layer_names=['repu']),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer.engine,
                     log_handler=ActivationsScalarHandler(model, reduction=torch.mean,
                                                          layer_names=['batch_norm', 'repu']),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer.engine,
                     log_handler=ActivationsScalarHandler(model, reduction=torch.std,
                                                          layer_names=['batch_norm', 'repu']),
                     event_name=Events.ITERATION_COMPLETED)

    # We need to close the logger with we are done
    tb_logger.close()

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


def main(args: Namespace, params: dict, dataset_name, run_postfix=""):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'heb-{}-{}'.format(dataset_name, identifier)
    if run_postfix:
        run += '-' + run_postfix

    # Loading the model and possibly initial weights
    model = models.create_fc1_model([28 ** 2, 2000], batch_norm=False)
    if args.initial_weights is not None:
        model = utils.load_weights(model, os.path.join(PATH, args.initial_weights))
        freeze_layers = ['linear1']
    else:
        freeze_layers = None

    # Device selection
    device = utils.get_device(args.device)
    print("Device set to '{}'.".format(device))
    model.to(device)

    # Data loaders
    train_loader, val_loader = data.get_data(params, dataset_name)

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run, log_dir=args.log_dir)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the learning rule, optimizer, learning rate scheduler, evaluator and trainer
    epochs = params['epochs']
    learning_rule = KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm'])
    optimizer = Local(named_params=model.named_parameters(), lr=params['lr'])
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
        h_es_handler.logger.setLevel(logging.DEBUG)
        h_evaluator.engine.add_event_handler(Events.COMPLETED, h_es_handler)

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

    # Handlers
    attach_handlers(run, model, optimizer, lr_scheduler, trainer, evaluator, visualizer, params)

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=epochs, eval_every=500)

    # # Save the final parameters with its corresponding metrics
    # visualizer.writer.add_hparams(params, evaluator.engine.state.metrics)
    #
    # return evaluator.engine.state.metrics
    return 0


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
    args_ = parser.parse_args()

    with open(os.path.join(PATH, args_.json)) as f:
        params_ = json.load(f)['params']

    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.DEBUG if args_.debug else logging.INFO)

    logging.debug("Arguments: {}.".format(vars(args_)))
    logging.debug("Parameters: {}.".format(params_))

    dataset_name_ = 'mnist'

    # param_name = "delta"
    # param_range = [0.6, 0.7, 0.8]
    # for param in param_range:
    #     params_[param_name] = param
    #
    #     print("Investigating {}={}...".format(param_name, param))
    #     metrics = main(args_, params_, run_postfix="d{}".format(param))
    #     print(metrics)

    main(args_, params_, dataset_name=dataset_name_)

import copy
import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace

import torch
from ignite.contrib.handlers import LRScheduler, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OptimizerParamsHandler
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping

import data
import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import HebbianEvaluator, SupervisedEvaluator
from pytorch_hebbian.handlers.tensorboard_logger import WeightsImageHandler
from pytorch_hebbian.handlers.tqdm_logger import TqdmLogger
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.metrics import UnitConvergence
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer, SupervisedTrainer

PATH = os.path.dirname(os.path.abspath(__file__))


def attach_handlers(run, model, optimizer, learning_rule, trainer, evaluator, train_loader, val_loader, params):
    # Metrics
    UnitConvergence(model[0], learning_rule.norm).attach(trainer.engine, 'unit_conv')

    # Tqdm logger
    pbar = ProgressBar(persist=True, bar_format=config.IGNITE_BAR_FORMAT)
    pbar.attach(trainer.engine, metric_names='all')
    tqdm_logger = TqdmLogger(pbar=pbar)
    # noinspection PyTypeChecker
    tqdm_logger.attach_output_handler(
        evaluator.engine,
        event_name=Events.COMPLETED,
        tag="validation",
        global_step_transform=global_step_from_engine(trainer.engine),
    )

    # Evaluator
    evaluator.attach(trainer.engine, Events.EPOCH_COMPLETED(every=100), train_loader, val_loader)

    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                     lr_lambda=lambda epoch: 1 - epoch / params['epochs'])
    lr_scheduler = LRScheduler(lr_scheduler)
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    # Early stopping
    mc_handler = ModelCheckpoint(config.MODELS_DIR, run.replace('/', '-'), n_saved=1, create_dir=True,
                                 require_empty=False,
                                 global_step_transform=global_step_from_engine(trainer.engine))
    trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, mc_handler, {'m': model})

    # Create a TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=os.path.join(config.TENSORBOARD_DIR, run))
    images, labels = next(iter(train_loader))
    tb_logger.writer.add_graph(copy.deepcopy(model).cpu(), images)
    tb_logger.writer.add_hparams(params, {})

    # noinspection PyTypeChecker
    tb_logger.attach_output_handler(
        evaluator.engine,
        event_name=Events.COMPLETED,
        tag="validation",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer.engine),
    )
    # noinspection PyTypeChecker
    tb_logger.attach_output_handler(
        trainer.engine,
        event_name=Events.EPOCH_COMPLETED,
        tag="train",
        metric_names=["unit_conv"]
    )
    input_shape = tuple(next(iter(train_loader))[0].shape[1:])
    tb_logger.attach(trainer.engine,
                     log_handler=WeightsImageHandler(model, input_shape),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer.engine, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.EPOCH_STARTED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=WeightsScalarHandler(model, layer_names=['linear1', 'linear2']),
    #                  event_name=Events.EPOCH_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=WeightsHistHandler(model, layer_names=['linear1', 'linear2']),
    #                  event_name=Events.EPOCH_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=ActivationsHistHandler(model, layer_names=['batch_norm', 'repu']),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=NumActivationsScalarHandler(model, layer_names=['repu']),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=ActivationsScalarHandler(model, reduction=torch.mean,
    #                                                       layer_names=['batch_norm', 'repu']),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=ActivationsScalarHandler(model, reduction=torch.std,
    #                                                       layer_names=['batch_norm', 'repu']),
    #                  event_name=Events.ITERATION_COMPLETED)

    return tb_logger


def main(args: Namespace, params: dict, dataset_name, run_postfix=""):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = '{}/heb/{}'.format(dataset_name, identifier)
    if run_postfix:
        run += '-' + run_postfix
    print("Starting run '{}'".format(run))

    # Loading the model and optionally initial weights for transfer learning
    model = models.create_conv1_model(28, 1, num_kernels=400, n=1, batch_norm=True)
    if args.initial_weights is not None:
        model = utils.load_weights(model, os.path.join(PATH, args.initial_weights))
        freeze_layers = ['linear1']
    else:
        freeze_layers = None

    # Device selection
    device = utils.get_device(args.device)
    model.to(device)
    print("Device set to '{}'.".format(device))

    # Data loaders
    train_loader, val_loader = data.get_data(params, dataset_name, subset=10000)

    # Creating the learning rule, optimizer, evaluator and trainer
    learning_rule = KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm'], normalize=False)
    optimizer = Local(named_params=model.named_parameters(), lr=params['lr'])

    # Initialization function called before each evaluation run of the Hebbian evaluator
    def init_function(h_model):
        h_criterion = torch.nn.CrossEntropyLoss()
        h_evaluator = SupervisedEvaluator(model=h_model, criterion=h_criterion, device=device)
        h_train_evaluator = SupervisedEvaluator(model=h_model, criterion=h_criterion, device=device)
        h_optimizer = torch.optim.Adam(params=h_model.parameters(), lr=1e-3)
        h_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(h_optimizer, 'max', verbose=True, patience=5,
                                                                    factor=0.5)
        h_trainer = SupervisedTrainer(model=h_model, optimizer=h_optimizer, criterion=h_criterion, device=device)

        # Tqdm logger
        h_pbar = ProgressBar(persist=False, bar_format=config.IGNITE_BAR_FORMAT)
        h_pbar.attach(h_trainer.engine, metric_names='all')
        h_tqdm_logger = TqdmLogger(pbar=h_pbar)
        # noinspection PyTypeChecker
        h_tqdm_logger.attach_output_handler(
            h_evaluator.engine,
            event_name=Events.COMPLETED,
            tag="validation",
            global_step_transform=global_step_from_engine(h_trainer.engine),
        )
        # noinspection PyTypeChecker
        h_tqdm_logger.attach_output_handler(
            h_train_evaluator.engine,
            event_name=Events.COMPLETED,
            tag="train",
            global_step_transform=global_step_from_engine(h_trainer.engine),
        )

        # Learning rate scheduling
        # The PyTorch Ignite LRScheduler class does not work with ReduceLROnPlateau
        h_evaluator.engine.add_event_handler(Events.COMPLETED,
                                             lambda engine: h_lr_scheduler.step(engine.state.metrics['accuracy']))

        # Model checkpoints
        h_handler = ModelCheckpoint(config.MODELS_DIR, run.replace('/', '-'), n_saved=1, create_dir=True,
                                    require_empty=False, score_name='acc',
                                    score_function=lambda engine: engine.state.metrics['accuracy'],
                                    global_step_transform=global_step_from_engine(trainer.engine))
        h_evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, h_handler, {'m': model})

        # Early stopping
        h_es_handler = EarlyStopping(patience=15,
                                     min_delta=0.0001,
                                     score_function=lambda engine: engine.state.metrics['accuracy'],
                                     trainer=h_trainer.engine, cumulative_delta=True)
        h_es_handler.logger.setLevel(logging.DEBUG)
        h_evaluator.engine.add_event_handler(Events.COMPLETED, h_es_handler)

        return h_trainer, h_train_evaluator, h_evaluator

    evaluator = HebbianEvaluator(model=model, score_name='accuracy',
                                 score_function=lambda engine: engine.state.metrics['accuracy'], epochs=500,
                                 init_function=init_function, supervised_from=-1)
    trainer = HebbianTrainer(model=model, learning_rule=learning_rule, optimizer=optimizer, supervised_from=-1,
                             freeze_layers=freeze_layers, device=device)

    # Handlers
    tb_logger = attach_handlers(run, model, optimizer, learning_rule, trainer, evaluator, train_loader, val_loader,
                                params)

    # Running the trainer
    trainer.run(train_loader=train_loader, epochs=params['epochs'])

    # Close the TensorBoard logger
    tb_logger.close()


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
    # parser.add_argument('--gpu_metrics', action='store_true', default=False,
    #                     help='enable gpu metrics in the trainer progress bar')
    args_ = parser.parse_args()

    with open(os.path.join(PATH, args_.json)) as f:
        params_ = json.load(f)['params']

    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.DEBUG if args_.debug else logging.INFO)

    logging.debug("Arguments: {}.".format(vars(args_)))
    logging.debug("Parameters: {}.".format(params_))

    main(args_, params_, dataset_name='mnist-fashion')

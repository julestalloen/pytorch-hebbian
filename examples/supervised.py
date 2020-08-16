import copy
import logging
import os
import time

import torch
from ignite.contrib.handlers import global_step_from_engine, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OptimizerParamsHandler
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint

import data
import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import SupervisedEvaluator
from pytorch_hebbian.handlers.tensorboard_logger import *
from pytorch_hebbian.handlers.tqdm_logger import TqdmLogger
from pytorch_hebbian.trainers import SupervisedTrainer

PATH = os.path.dirname(os.path.abspath(__file__))


def attach_handlers(run, model, optimizer, trainer, train_evaluator, evaluator, train_loader, val_loader,
                    params):
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
    # noinspection PyTypeChecker
    tqdm_logger.attach_output_handler(
        train_evaluator.engine,
        event_name=Events.COMPLETED,
        tag="train",
        global_step_transform=global_step_from_engine(trainer.engine),
    )

    # Evaluators
    train_evaluator.attach(trainer.engine, Events.EPOCH_COMPLETED, train_loader)
    evaluator.attach(trainer.engine, Events.EPOCH_COMPLETED, data=val_loader)

    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5, factor=0.5)
    evaluator.engine.add_event_handler(Events.COMPLETED,
                                       lambda engine: lr_scheduler.step(engine.state.metrics['accuracy']))

    # Early stopping
    es_handler = EarlyStopping(patience=15, score_function=lambda engine: engine.state.metrics['accuracy'],
                               trainer=trainer.engine, cumulative_delta=True, min_delta=0.0001)
    if 'train_all' in params and params['train_all']:
        train_evaluator.engine.add_event_handler(Events.COMPLETED, es_handler)
    else:
        evaluator.engine.add_event_handler(Events.COMPLETED, es_handler)

    es_handler.logger.setLevel(logging.DEBUG)

    # Model checkpoints
    name = run.replace('/', '-')
    mc_handler = ModelCheckpoint(config.MODELS_DIR, name, n_saved=1, create_dir=True, require_empty=False,
                                 score_name='acc', score_function=lambda engine: engine.state.metrics['accuracy'],
                                 global_step_transform=global_step_from_engine(trainer.engine))
    evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, mc_handler, {'m': model})

    # TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=os.path.join(config.TENSORBOARD_DIR, run))
    images, labels = next(iter(train_loader))
    tb_logger.writer.add_graph(copy.deepcopy(model).cpu(), images)
    tb_logger.writer.add_hparams(params, {'hparam/dummy': 0})

    # noinspection PyTypeChecker
    tb_logger.attach_output_handler(
        train_evaluator.engine,
        event_name=Events.COMPLETED,
        tag="train",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer.engine),
    )
    # noinspection PyTypeChecker
    tb_logger.attach_output_handler(
        evaluator.engine,
        event_name=Events.COMPLETED,
        tag="validation",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer.engine),
    )
    input_shape = tuple(next(iter(train_loader))[0].shape[1:])
    tb_logger.attach(trainer.engine,
                     log_handler=WeightsImageHandler(model, input_shape),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer.engine, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.EPOCH_STARTED)
    # tb_logger.attach(trainer.engine, log_handler=WeightsScalarHandler(model), event_name=Events.EPOCH_COMPLETED)
    # tb_logger.attach(trainer.engine, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=ActivationsHistHandler(model, layer_names=['linear1', 'batch_norm', 'repu']),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=NumActivationsScalarHandler(model, layer_names=['linear1', 'repu']),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=ActivationsScalarHandler(model, reduction=torch.mean,
    #                                                       layer_names=['linear1', 'batch_norm', 'repu']),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(trainer.engine,
    #                  log_handler=ActivationsScalarHandler(model, reduction=torch.std,
    #                                                       layer_names=['linear1', 'batch_norm', 'repu']),
    #                  event_name=Events.ITERATION_COMPLETED)

    return es_handler, tb_logger


def main(params, dataset_name, transfer_learning=False):
    # # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = '{}/sup/{}'.format(dataset_name, identifier)
    if transfer_learning:
        run += "-tl"
    if 'train_all' in params and params['train_all']:
        run += "-test"
    print("Starting run '{}'".format(run))

    # Loading the model and optionally initial weights for transfer learning
    model = models.create_conv1_model(28, 1, num_kernels=400, n=1, batch_norm=True)
    if transfer_learning:
        weights_path = "../output/models/heb-mnist-fashion-20200607-015911_m_100_acc=0.855.pth"
        model = utils.load_weights(model, os.path.join(PATH, weights_path), layer_names=['conv1'], freeze=True)

    # Device selection
    device = utils.get_device()
    model.to(device)
    print("Device set to '{}'.".format(device))

    # Data loaders
    train_loader, val_loader = data.get_data(params, dataset_name, subset=params['train_subset'])

    # Creating the criterion, optimizer, optimizer, evaluators and trainer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    train_evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

    es_handler, tb_logger = attach_handlers(run, model, optimizer, trainer, train_evaluator, evaluator, train_loader,
                                            val_loader, params)

    # Running the trainer
    trainer.run(train_loader=train_loader, epochs=params['epochs'])

    # Save the best score and close the TensorBoard logger
    tb_logger.writer.add_hparams(params, {"hparam/accuracy": es_handler.best_score})
    tb_logger.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.INFO)

    params_ = {
        'train_batch_size': 256,
        'val_batch_size': 256,
        'val_split': 0.2,
        'epochs': 500,
        'lr': 1e-3,
        'train_subset': 20000,
        'train_all': False,
    }

    main(params_, dataset_name='mnist-fashion', transfer_learning=True)

import logging
import os
import time

import torch
from ignite.contrib.handlers.base_logger import global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OptimizerParamsHandler
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint

import data
import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import SupervisedEvaluator
from pytorch_hebbian.handlers.tensorboard_logger import *
from pytorch_hebbian.trainers import SupervisedTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def attach_handlers(run, model, optimizer, lr_scheduler, trainer, train_evaluator, evaluator, visualizer, params):
    # Learning rate scheduling
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
    tb_logger.writer = visualizer.writer
    tb_logger.attach(trainer.engine, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.EPOCH_STARTED)
    tb_logger.attach(trainer.engine, log_handler=WeightsScalarHandler(model), event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer.engine, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
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
    tb_logger.close()


def main(params, dataset_name, transfer_learning=False):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'sup/{}/{}'.format(dataset_name, identifier)
    if transfer_learning:
        run += "-tl"
    if 'train_all' in params and params['train_all']:
        run += "-test"

    # Loading the model and possibly initial weights
    model = models.create_fc1_model(hu=[28 ** 2, 2000], n=1.5, batch_norm=True)
    if transfer_learning:
        weights_path = "../output/models/heb-mnist-fashion-20200522-174314_m_499.pth"
        model = utils.load_weights(model, os.path.join(PATH, weights_path), layer_names=['linear1'], freeze=True)

    # Data loaders
    train_loader, val_loader = data.get_data(params, dataset_name, subset=params['train_subset'])

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the criterion, optimizer, optimizer, evaluator and trainer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5, factor=0.5)
    train_evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion, train_evaluator=train_evaluator,
                                evaluator=evaluator, visualizer=visualizer)

    attach_handlers(run, model, optimizer, lr_scheduler, trainer, train_evaluator, evaluator, visualizer, params)

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=params['epochs'], eval_every=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.INFO)

    params_ = {
        'train_batch_size': 128,
        'val_batch_size': 128,
        'val_split': 0.2,
        'epochs': 500,
        'lr': 1e-3,
        'train_subset': 30000,
        'train_all': True,
    }

    main(params_, dataset_name='mnist-fashion', transfer_learning=True)

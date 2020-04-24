import logging
import os
import time

import torch
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import SupervisedEvaluator
from pytorch_hebbian.trainers import SupervisedTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def main(params):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'sup-{}'.format(identifier)
    logging.info("Starting run '{}'.".format(run))

    if params['train_all']:
        logging.info('Training on all train data!')

    # Loading the model and possibly initial weights
    model = models.dense_net1_mnist
    weights_path = "../output/models/heb-20200417-134912_m_1000_acc=0.8381666666666666.pth"
    model = utils.load_weights(model, os.path.join(PATH, weights_path), layer_names=['1'], freeze=True)

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = Subset(dataset, [i for i in range(10000)])

    if params['train_all']:
        train_loader = DataLoader(dataset, batch_size=params['train_batch_size'], shuffle=True)
        val_loader = None
    else:
        train_dataset, val_dataset = utils.split_dataset(dataset, val_split=params['val_split'])
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the criterion, optimizer, optimizer, evaluator and trainer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=10, factor=0.5)
    train_evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    if params['train_all']:
        evaluator = None
    else:
        evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion, train_evaluator=train_evaluator,
                                evaluator=evaluator, visualizer=visualizer)

    if params['train_all']:
        eval_to_monitor = train_evaluator
    else:
        eval_to_monitor = evaluator

    # Learning rate scheduling
    eval_to_monitor.engine.add_event_handler(Events.COMPLETED,
                                             lambda engine: lr_scheduler.step(engine.state.metrics['accuracy']))

    # Early stopping
    es_handler = EarlyStopping(patience=30, score_function=lambda engine: engine.state.metrics['accuracy'],
                               trainer=trainer.engine, cumulative_delta=True, min_delta=0.0001)
    eval_to_monitor.engine.add_event_handler(Events.COMPLETED, es_handler)
    es_handler.logger.setLevel(logging.DEBUG)

    # Model checkpoints
    mc_handler = ModelCheckpoint(config.MODELS_DIR, run, n_saved=1, create_dir=True, require_empty=False,
                                 score_name='acc', score_function=lambda engine: engine.state.metrics['accuracy'],
                                 global_step_transform=global_step_from_engine(trainer.engine))
    eval_to_monitor.engine.add_event_handler(Events.EPOCH_COMPLETED, mc_handler, {'m': model})

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=params['epochs'], eval_every=1)

    if not params['train_all']:
        # Save the final parameters with its corresponding metrics
        visualizer.writer.add_hparams(params, {'hparam/accuracy': es_handler.best_score})

    return es_handler.best_score


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.INFO)

    params_ = {
        'train_batch_size': 256,
        'val_batch_size': 256,
        'val_split': 0.2,
        'epochs': 500,
        'lr': 1e-4,
        "train_all": False,
    }

    # results = []
    # param_range = [1]  # , 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    # for n in param_range:
    #     print("Currently evaluating n={}.".format(n))
    #     params_['n'] = n
    #     result = main(params_)
    #     results.append(result)
    #     print("Result={}.".format(result))
    #
    # print(results)

    main(params_)

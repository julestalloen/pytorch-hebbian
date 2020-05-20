import logging
import os
import time

import torch

import data
import models
from pytorch_hebbian import config, utils
from pytorch_hebbian import nn
from pytorch_hebbian.evaluators import SupervisedEvaluator
from pytorch_hebbian.trainers import SupervisedTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer
from supervised import attach_handlers

PATH = os.path.dirname(os.path.abspath(__file__))


def main(params, dataset_name, transfer_learning=False):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'sup-{}-{}'.format(dataset_name, identifier)
    if transfer_learning:
        run += "-tl"

    # Loading the model and possibly initial weights
    model = models.create_fc1_model([28 ** 2, 2000], n=4.5, batch_norm=False)
    if transfer_learning:
        weights_path = "../output/models/heb-20200408-193344_m_1000_acc=0.929.pth"
        model = utils.load_weights(model, os.path.join(PATH, weights_path), layer_names=[('1', 'linear1')], freeze=True)

    # Custom initialization for final layer
    dict(model.named_children())['linear2'].weight.data.normal_(mean=0, std=0.2)

    # Data loaders
    train_loader, val_loader = data.get_data(params, dataset_name, subset=10000)

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the criterion, optimizer, optimizer, evaluator and trainer
    criterion = nn.SPELoss(m=8, beta=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=10, factor=0.5)
    train_evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion, train_evaluator=train_evaluator,
                                evaluator=evaluator, visualizer=visualizer)

    attach_handlers(run, model, optimizer, lr_scheduler, trainer, evaluator, visualizer, params)

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=params['epochs'], eval_every=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.INFO)

    params_ = {
        'train_batch_size': 100,
        'val_batch_size': 100,
        'val_split': 0.2,
        'epochs': 500,
        'lr': 1e-3,
        "train_all": False,
    }

    main(params_, dataset_name='mnist', transfer_learning=True)

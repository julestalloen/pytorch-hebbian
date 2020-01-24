import logging
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from model import Net
from pytorch_hebbian.evaluators import HebbianEvaluator
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer
from pytorch_hebbian.utils.data import split_dataset
from pytorch_hebbian.utils.tensorboard import write_stats
from pytorch_hebbian.visualizers import TensorBoardVisualizer


def main(params):
    run = 'hebbian-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    logging.info("Starting run '{}'.".format(run))

    model = Net([params['input_size'], params['hidden_units'], params['output_size']])

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    train_dataset, val_dataset = split_dataset(dataset, val_split=params['val_split'])
    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # TensorBoard visualizer
    visualizer = TensorBoardVisualizer(run=run)
    # Write some basis stats
    write_stats(visualizer, model, train_loader, params)

    epochs = params['epochs']
    learning_rule = KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm'])
    optimizer = Local(params=model.parameters(), lr=params['lr'])
    # noinspection PyTypeChecker
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs)
    evaluator = HebbianEvaluator(model=model, data_loader=val_loader)
    trainer = HebbianTrainer(model=model,
                             learning_rule=learning_rule,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             evaluator=evaluator,
                             visualizer=visualizer)
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=epochs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    params_mnist = {
        'input_size': 28 ** 2,
        'hidden_units': 400,
        'output_size': 10,
        'train_batch_size': 1000,
        'val_batch_size': 64,
        'val_split': 0.2,
        'epochs': 100,
        'delta': 0.4,
        'k': 7,
        'norm': 3,
        'lr': 0.04
    }

    params_cifar = {
        'input_size': 32 ** 2 * 3,
        'hidden_units': 100,
        'output_size': 10,
        'train_batch_size': 1000,
        'val_batch_size': 64,
        'val_split': 0.2,
        'epochs': 1000,
        'delta': 0.2,
        'k': 2,
        'norm': 5,
        'lr': 0.02
    }

    main(params_mnist)

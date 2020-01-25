import json
import logging
import os
import time
from argparse import ArgumentParser

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
    evaluator = HebbianEvaluator(model=model)
    trainer = HebbianTrainer(model=model,
                             learning_rule=learning_rule,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             evaluator=evaluator,
                             visualizer=visualizer)
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=epochs, eval_every=20)

    # Save the params with metrics
    visualizer.writer.add_hparams(params, evaluator.metrics)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    # TODO: WIP
    parser = ArgumentParser()
    parser.add_argument('--json', type=str,
                        help='use a preset json file to specify parameters')
    # parser.add_argument('--batch_size', type=int, default=64,
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--val_batch_size', type=int, default=1000,
    #                     help='input batch size for validation (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
    #                     help="log directory for Tensorboard log output")
    args = parser.parse_args()

    if args.json is not None:
        with open(os.path.join(config.PARAMS_DIR, args.json)) as f:
            params_ = json.load(f)['params']
        logging.info("Loaded parameters from '{}'.".format(args.json))

    main(params_)

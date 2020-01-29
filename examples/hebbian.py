import json
import logging
import os
import time
from argparse import ArgumentParser

import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from examples.model import Net
from pytorch_hebbian import config
from pytorch_hebbian.evaluators import HebbianEvaluator
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer
from pytorch_hebbian.utils.data import split_dataset
from pytorch_hebbian.utils.tensorboard import write_stats
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def main(params):
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'heb-{}'.format(identifier)
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

    # Model checkpoint saving
    handler = ModelCheckpoint(config.MODELS_DIR, 'heb-' + identifier, n_saved=2, create_dir=True, require_empty=False,
                              score_name='loss', score_function=lambda engine: -engine.state.metrics['loss'],
                              global_step_transform=global_step_from_engine(trainer.engine))
    evaluator.evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, handler, {'m': model})

    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=epochs, eval_every=50)

    # Save the params with metrics
    visualizer.writer.add_hparams(params, evaluator.metrics)


if __name__ == '__main__':
    # TODO: WIP
    parser = ArgumentParser()
    parser.add_argument('--json', type=str,
                        help='use a preset json file to specify parameters')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug logging')
    parser.add_argument('--hidden_units', type=int,
                        help='number of hidden units of the model')
    # parser.add_argument('--batch_size', type=int, default=64,
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--val_batch_size', type=int, default=1000,
    #                     help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train')
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
    #                     help="log directory for Tensorboard log output")
    args = parser.parse_args()

    if args.json is not None:
        with open(os.path.join(PATH, args.json)) as f:
            params_ = json.load(f)['params']
            set_args = {k: v for k, v in vars(args).items() if v is not None}
            params_.update(set_args)
    else:
        params_ = vars(args)
    del params_['debug']

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    if args.json is not None:
        logging.info("Loaded parameters from '{}'.".format(args.json))
    logging.debug("Parameters: {}.".format(params_))

    main(params_)
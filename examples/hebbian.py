import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace

import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
from pytorch_hebbian import config
from pytorch_hebbian.evaluators import HebbianEvaluator
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer
from pytorch_hebbian.utils.data import split_dataset
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def load_weights(model, weights):
    state_dict_path = os.path.join(PATH, weights)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(device)))
    logging.info("Loaded initial model weights from '{}'.".format(weights))

    return model


def main(args: Namespace, params: dict):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'heb-{}'.format(identifier)
    logging.info("Starting run '{}'.".format(run))

    # Loading the model and possibly initial weights
    model = models.dense_net
    if args.initial_weights is not None:
        initial_weights = args.initial_weights
        model = load_weights(model, initial_weights)

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    train_dataset, val_dataset = split_dataset(dataset, val_split=params['val_split'])
    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run, log_dir=args.log_dir)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the learning rule, optimizer, learning rate scheduler, evaluator and trainer
    epochs = params['epochs']
    learning_rule = KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm'])
    optimizer = Local(named_params=model.named_parameters(), lr=params['lr'])
    # TODO: Typing issue should be fixed in future pytorch update
    #   https://github.com/pytorch/pytorch/issues/32645
    # noinspection PyTypeChecker
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs)
    evaluator = HebbianEvaluator(model=model, epochs=500)
    trainer = HebbianTrainer(model=model,
                             learning_rule=learning_rule,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             evaluator=evaluator,
                             visualizer=visualizer)

    # Adding handlers for model checkpoints and visualizing to the engine
    handler = ModelCheckpoint(config.MODELS_DIR, 'heb-' + identifier, n_saved=1, create_dir=True, require_empty=False,
                              score_name='loss', score_function=lambda engine: -engine.state.metrics['loss'],
                              global_step_transform=global_step_from_engine(trainer.engine))
    evaluator.evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, handler, {'m': model})

    @trainer.engine.on(Events.EPOCH_STARTED)
    def log_learning_rate(engine):
        visualizer.writer.add_scalar('learning_rate', trainer.lr_scheduler.get_param(), engine.state.epoch - 1)

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=epochs, eval_every=50)

    # Save the final parameters with its corresponding metrics
    visualizer.writer.add_hparams(params, evaluator.metrics)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, required=True,
                        help='use a preset json file to specify parameters')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enable debug logging')
    # parser.add_argument('--train_batch_size', type=int, default=128,
    #                     help='batch size for training')
    # parser.add_argument('--val_batch_size', type=int, default=128,
    #                     help='batch size for validation')
    # parser.add_argument('--val_split', type=int, default=0.2,
    #                     help='percentage of the data to use for validation')
    # parser.add_argument('--epochs', type=int, default=1000,
    #                     help='number of epochs to train')
    # parser.add_argument('--delta', type=float, default=0.4,
    #                     help='delta parameter of the Krotov learning rule.')
    # parser.add_argument('--k', type=int, default=7,
    #                     help='k parameter of the Krotov learning rule.')
    # parser.add_argument('--norm', type=int, default=3,
    #                     help='norm parameter of the Krotov learning rule.')
    # parser.add_argument('--lr', type=float, default=0.04,
    #                     help='learning rate')
    parser.add_argument("--log_dir", type=str, default=config.TENSORBOARD_DIR,
                        help="log directory for Tensorboard log output")
    parser.add_argument('--initial_weights', type=str,
                        help='model weights to initialize training')
    args_ = parser.parse_args()

    with open(os.path.join(PATH, args_.json)) as f:
        params_ = json.load(f)['params']

    # TODO: why is this needed?
    logging.root.handlers = []
    logging.basicConfig(level=logging.DEBUG if args_.debug else logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    logging.debug("Arguments: {}.".format(vars(args_)))
    logging.debug("Parameters: {}.".format(params_))

    main(args_, params_)

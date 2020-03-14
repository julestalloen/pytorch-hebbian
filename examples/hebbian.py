import json
import logging
import os
import time
from argparse import ArgumentParser, Namespace

# import numpy as np
# from matplotlib import pyplot as plt
# from scipy import signal
import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.evaluators import HebbianEvaluator
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def load_weights(model, weights):
    state_dict_path = os.path.join(PATH, weights)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=False)

    logging.info("Loaded initial model weights from '{}'.".format(weights))

    return model


def main(args: Namespace, params: dict):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'heb-{}'.format(identifier)
    logging.info("Starting run '{}'.".format(run))

    # Loading the model and possibly initial weights
    model = models.conv_net2
    if args.initial_weights is not None:
        initial_weights = args.initial_weights
        model = load_weights(model, initial_weights)
        freeze_layers = ['0']
    else:
        freeze_layers = None

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    dataset = Subset(dataset, [i for i in range(10000)])
    train_dataset, val_dataset = utils.split_dataset(dataset, val_split=params['val_split'])
    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run, log_dir=args.log_dir)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the learning rule, optimizer, learning rate scheduler, evaluator and trainer
    epochs = params['epochs']
    learning_rule = {
        '0': KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm']),
        '3': KrotovsRule(delta=0.2, k=params['k'], norm=params['norm']),
    }
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
                             supervised_from=-1,
                             freeze_layers=freeze_layers,
                             evaluator=evaluator,
                             visualizer=visualizer)

    # Adding handlers for model checkpoints and visualizing to the evaluator and trainer engine
    handler = ModelCheckpoint(config.MODELS_DIR, 'heb-' + identifier, n_saved=1, create_dir=True, require_empty=False,
                              score_name='loss', score_function=lambda engine: -engine.state.metrics['loss'],
                              global_step_transform=global_step_from_engine(trainer.engine))
    evaluator.evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, handler, {'m': model})

    @trainer.engine.on(Events.EPOCH_STARTED)
    def log_learning_rate(engine):
        visualizer.writer.add_scalar('learning_rate', trainer.lr_scheduler.get_param(), engine.state.epoch - 1)

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

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=epochs, eval_every=10)

    # Save the final parameters with its corresponding metrics
    visualizer.writer.add_hparams(params, evaluator.metrics)


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
    args_ = parser.parse_args()

    with open(os.path.join(PATH, args_.json)) as f:
        params_ = json.load(f)['params']

    logging.basicConfig(level=logging.DEBUG if args_.debug else logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    logging.debug("Arguments: {}.".format(vars(args_)))
    logging.debug("Parameters: {}.".format(params_))

    main(args_, params_)

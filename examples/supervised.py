import logging
import time

import torch
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from examples.model import Net
from pytorch_hebbian import config
from pytorch_hebbian.evaluators import SupervisedEvaluator
from pytorch_hebbian.trainers import SupervisedTrainer
from pytorch_hebbian.utils.data import split_dataset
from pytorch_hebbian.utils.tensorboard import write_stats
from pytorch_hebbian.visualizers import TensorBoardVisualizer


def main(params):
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'sup-{}'.format(identifier)
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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion, evaluator=evaluator,
                                visualizer=visualizer)

    # Early stopping
    handler = EarlyStopping(patience=5, score_function=lambda engine: -engine.state.metrics['loss'],
                            trainer=trainer.engine, cumulative_delta=True)
    evaluator.engine.add_event_handler(Events.COMPLETED, handler)

    # Model checkpoint saving
    handler = ModelCheckpoint(config.MODELS_DIR, 'sup-' + identifier, n_saved=2, create_dir=True, require_empty=False,
                              score_name='loss', score_function=lambda engine: -engine.state.metrics['loss'],
                              global_step_transform=global_step_from_engine(trainer.engine))
    evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, handler, {'m': model})

    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=params['epochs'], eval_every=2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    params_ = {
        'input_size': 28 ** 2,
        'hidden_units': 400,
        'output_size': 10,
        'train_batch_size': 100,
        'val_batch_size': 100,
        'val_split': 0.2,
        'epochs': 100,
        'lr': 0.001
    }

    main(params_)

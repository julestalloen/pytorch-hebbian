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
from pytorch_hebbian.utils import load_weights
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def main(params):
    # Creating an identifier for this run
    identifier = time.strftime("%Y%m%d-%H%M%S")
    run = 'sup-{}'.format(identifier)
    logging.info("Starting run '{}'.".format(run))

    # Loading the model and possibly initial weights
    model = models.dense_net1
    weights_path = "models/heb-20200408-155246_m_500_acc=0.929-1.weight.pth"
    state_dict_path = os.path.join(PATH, weights_path)
    model = load_weights(model, state_dict_path)
    for param in list(model.children())[1].parameters():
        param.requires_grad = False
    print("Freezed", list(model.children())[1])

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = Subset(dataset, [i for i in range(10000)])
    train_dataset, val_dataset = utils.split_dataset(dataset, val_split=params['val_split'])
    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # Creating the TensorBoard visualizer and writing some initial statistics
    visualizer = TensorBoardVisualizer(run=run)
    visualizer.visualize_stats(model, train_loader, params)

    # Creating the criterion, optimizer, optimizer, evaluator and trainer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=4, factor=0.2)
    train_evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    evaluator = SupervisedEvaluator(model=model, criterion=criterion)
    trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion, train_evaluator=train_evaluator,
                                evaluator=evaluator, visualizer=visualizer)

    # Learning rate scheduling
    evaluator.engine.add_event_handler(Events.COMPLETED,
                                       lambda engine: lr_scheduler.step(engine.state.metrics['accuracy']))

    # Early stopping
    handler = EarlyStopping(patience=10, score_function=lambda engine: engine.state.metrics['accuracy'],
                            trainer=trainer.engine, cumulative_delta=True)
    evaluator.engine.add_event_handler(Events.COMPLETED, handler)
    handler.logger.setLevel(logging.INFO)

    # Model checkpoints
    handler = ModelCheckpoint(config.MODELS_DIR, run, n_saved=1, create_dir=True, require_empty=False,
                              score_name='acc', score_function=lambda engine: engine.state.metrics['accuracy'],
                              global_step_transform=global_step_from_engine(trainer.engine))
    evaluator.engine.add_event_handler(Events.EPOCH_COMPLETED, handler, {'m': model})

    # Running the trainer
    trainer.run(train_loader=train_loader, val_loader=val_loader, epochs=params['epochs'], eval_every=2,
                vis_weights_every=-1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    params_ = {
        'train_batch_size': 128,
        'val_batch_size': 128,
        'val_split': 0.2,
        'epochs': 100,
        'lr': 0.001
    }

    main(params_)

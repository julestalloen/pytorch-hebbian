import logging
import time

import torch
import torchvision
from torchvision import datasets, transforms

import config
from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator
from pytorch_hebbian.learning_engines.supervised_engine import SupervisedEngine
from pytorch_hebbian.utils.visualization import plot_learning_curve, plot_accuracy
from pytorch_hebbian.visualizers import TensorBoardVisualizer


# noinspection PyUnresolvedReferences
def main(params):
    run = 'supervised-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    logging.info("Starting run '{}'.".format(run))

    model = torch.nn.Sequential(
        torch.nn.Linear(params['input_size'], params['hidden_units']),
        torch.nn.ReLU(),
        torch.nn.Linear(params['hidden_units'], params['output_size']),
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    train_size = int((1 - params['val_split']) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=True)

    # TensorBoard visualizer
    visualizer = TensorBoardVisualizer(run=run)

    # Visualize some input samples and the hyperparameters
    images, labels = next(iter(train_loader))
    visualizer.writer.add_image('input/samples', torchvision.utils.make_grid(images[:64]))
    num_project = 128
    visualizer.project(images[:num_project], labels[:num_project], params['input_size'])
    visualizer.writer.add_hparams(params, {})

    epochs = params['epochs']
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    evaluator = SupervisedEvaluator(model=model, data_loader=val_loader, loss_criterion=criterion)
    learning_engine = SupervisedEngine(criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                       evaluator=evaluator)
    model = learning_engine.train(model=model, data_loader=train_loader, epochs=epochs, eval_every=1)

    # TODO: save model
    print(model)

    # Learning curves
    plot_learning_curve(learning_engine.losses, evaluator.losses)
    plot_accuracy(evaluator.accuracies)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)

    params_ = {
        'input_size': 28 ** 2,
        'hidden_units': 100,
        'output_size': 10,
        'train_batch_size': 64,
        'val_batch_size': 64,
        'val_split': 0.2,
        'epochs': 100,
        'lr': 0.001
    }

    main(params_)

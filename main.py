import logging

import torch
import torchvision
from torchvision import datasets, transforms

import config
from pytorch_hebbian.evaluators.hebbian_evaluator import HebbianEvaluator
from pytorch_hebbian.learning_engines.hebbian_engine import HebbianEngine
from pytorch_hebbian.learning_rules.krotov import KrotovsRule
from pytorch_hebbian.optimizers.local import Local
from pytorch_hebbian.utils.visualization import plot_learning_curve, plot_accuracy
from pytorch_hebbian.visualizers import PerceptronVisualizer, TensorBoardVisualizer


# noinspection PyTypeChecker,PyUnresolvedReferences
def main(params):
    model = torch.nn.Sequential(
        torch.nn.Linear(params['input_size'], params['hidden_units'], bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(params['hidden_units'], params['output_size']),
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=True)

    tb_visualizer = TensorBoardVisualizer()

    # Visualize some input samples
    images, labels = next(iter(train_loader))
    tb_visualizer.writer.add_image('initialization/some_input_samples', torchvision.utils.make_grid(images[:64]))
    tb_visualizer.project(images[:100], labels[:100], params['input_size'])

    epochs = params['epochs']
    learning_rule = KrotovsRule(delta=params['delta'], k=params['k'], norm=params['norm'])
    optimizer = Local(params=model.parameters(), lr=params['lr'])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs)
    evaluator = HebbianEvaluator(model=model, data_loader=val_loader)
    visualizer = PerceptronVisualizer()
    learning_engine = HebbianEngine(learning_rule=learning_rule,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    evaluator=evaluator,
                                    visualizer=visualizer)
    model = learning_engine.train(model=model, data_loader=train_loader, epochs=epochs,
                                  eval_every=50, checkpoint_every=None)

    print(model)

    # Learning curves
    plot_learning_curve(evaluator.supervised_engine.losses, evaluator.supervised_engine.evaluator.losses)
    plot_accuracy(evaluator.supervised_engine.evaluator.accuracies)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)

    params_mnist = {
        'input_size': 28 ** 2,
        'hidden_units': 400,
        'output_size': 10,
        'train_batch_size': 1000,
        'val_batch_size': 64,
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
        'epochs': 1000,
        'delta': 0.2,
        'k': 2,
        'norm': 5,
        'lr': 0.02
    }

    main(params_mnist)

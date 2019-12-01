import logging

import torch
import torchvision
from torchvision import datasets, transforms

import config
from pytorch_hebbian.utils.visualization import show_image
from pytorch_hebbian.learning_rules.krotov import KrotovsRule
from pytorch_hebbian.learning_engines.hebbian_engine import HebbianEngine
from pytorch_hebbian.optimizers.local import Local
from pytorch_hebbian.evaluators.hebbian_evaluator import HebbianEvaluator
from pytorch_hebbian.visualizers.weight_visualizer import PerceptronVisualizer


# noinspection PyTypeChecker
def main(params):
    model = torch.nn.Sequential(
        torch.nn.Linear(params['input_size'], params['hidden_units'], bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(params['hidden_units'], params['output_size']),
    )

    transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])
    # dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    # dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
    # TODO: create train val split
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = None

    # Visualize some input samples
    images, labels = next(iter(train_loader))
    show_image(torchvision.utils.make_grid(images[:64]), title='Some input samples')

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
                                  eval_every=5, checkpoint_every=10)

    print(model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(module)s:%(message)s')

    params_mnist = {
        'input_size': 28 ** 2,
        'hidden_units': 16,
        'output_size': 10,
        'batch_size': 100,
        'epochs': 100,
        'delta': 0.3,
        'k': 2,
        'norm': 2,
        'lr': 0.04
    }

    params_cifar = {
        'input_size': 32 ** 2 * 3,
        'hidden_units': 100,
        'output_size': 10,
        'batch_size': 1000,
        'epochs': 1000,
        'delta': 0.2,
        'k': 2,
        'norm': 5,
        'lr': 0.02
    }

    main(params_cifar)

import logging

import torch
import torchvision
from torchvision import datasets, transforms

import config
from pytorch_hebbian.utils.visualization import show_image, plot_learning_curve, plot_accuracy
from pytorch_hebbian.learning_engines.supervised_engine import SupervisedEngine
from pytorch_hebbian.evaluators.supervised_evaluator import SupervisedEvaluator


# noinspection PyTypeChecker,PyUnresolvedReferences
def main(params):
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
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True)

    # Visualize some input samples
    images, labels = next(iter(train_loader))
    show_image(torchvision.utils.make_grid(images[:64]), title='Some input samples')

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
        'batch_size': 64,
        'epochs': 100,
        'lr': 0.001
    }

    main(params_)

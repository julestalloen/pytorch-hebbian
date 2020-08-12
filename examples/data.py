import logging
import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from pytorch_hebbian import config, utils

PATH = os.path.dirname(os.path.abspath(__file__))


def get_data(params, dataset_name, subset=None):
    load_test = 'train_all' in params and params['train_all']
    test_dataset = None

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset_name == 'mnist':
        dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
        if load_test:
            test_dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, train=False,
                                                transform=transform)
    elif dataset_name == 'mnist-fashion':
        dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
        if load_test:
            test_dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, train=False,
                                                       transform=transform)
    elif dataset_name == "cifar-10":
        dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
        if load_test:
            test_dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, train=False,
                                                  transform=transform)
    else:
        raise AttributeError('Dataset not found')

    if subset is not None and subset > 0:
        dataset = Subset(dataset, random.sample(range(len(dataset)), subset))

    if load_test:
        train_loader = DataLoader(dataset, batch_size=params['train_batch_size'], shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], shuffle=False)
    else:
        train_dataset, val_dataset = utils.split_dataset(dataset, val_split=params['val_split'])
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # Analyze dataset
    data_batch = next(iter(train_loader))[0]
    logging.debug("Data batch min: {:.4f}, max: {:.4f}.".format(torch.min(data_batch),
                                                                torch.max(data_batch)))
    logging.debug("Data batch mean: {1:.4f}, std: {0:.4f}.".format(*torch.std_mean(data_batch)))

    return train_loader, val_loader

import logging
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
from pytorch_hebbian import config
from pytorch_hebbian.evaluators import SupervisedEvaluator
from pytorch_hebbian.utils import load_weights
from pytorch_hebbian.visualizers import TensorBoardVisualizer

PATH = os.path.dirname(os.path.abspath(__file__))


def main(params):
    # Loading the model and possibly initial weights
    model = models.dense_net1
    weights_path = "../output/models/sup-20200411-185539_m_80_acc=0.9895666666666667.pth"
    state_dict_path = os.path.join(PATH, weights_path)
    model = load_weights(model, state_dict_path)

    # Extracting the identifier for this run
    run = os.path.splitext(os.path.basename(weights_path))[0].split('_')[0]
    run += '/test'
    logging.info("Starting run '{}'.".format(run))

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform, train=False)
    test_loader = DataLoader(dataset, batch_size=params['val_batch_size'], shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    evaluator = SupervisedEvaluator(model=model, criterion=criterion)

    evaluator.run(test_loader)
    print(evaluator.metrics)

    # Save the final parameters with its corresponding metrics
    visualizer = TensorBoardVisualizer(run=run)
    visualizer.writer.add_hparams(params, evaluator.metrics)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    params_ = {
        'train_batch_size': 128,
        'val_batch_size': 128,
        'val_split': 0.2,
        'epochs': 500,
        'lr': 0.001
    }

    main(params_)

import logging
import random

from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

import models
from pytorch_hebbian import config, utils
from pytorch_hebbian.learning_rules import OjasRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer


def main():
    # Loading the model and possibly initial weights
    model = models.create_fc1_model([28 ** 2, 16])

    # Device selection
    device = utils.get_device()
    logging.info("Device set to '{}'.".format(device))
    model.to(device)

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    dataset = Subset(dataset, random.sample(range(len(dataset)), 1000))
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Creating the learning rule, optimizer and trainer
    learning_rule = OjasRule()
    optimizer = Local(named_params=model.named_parameters(), lr=0.1)
    visualizer = TensorBoardVisualizer(run='oja-test', log_dir=config.TENSORBOARD_DIR)
    trainer = HebbianTrainer(model=model, learning_rule=learning_rule, optimizer=optimizer, visualizer=visualizer,
                             device=device)

    # Running the trainer
    trainer.run(train_loader=train_loader, epochs=100, vis_weights_every=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.INFO)

    main()

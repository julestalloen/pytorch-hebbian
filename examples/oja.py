import logging
import os

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import models
from pytorch_hebbian import config
from pytorch_hebbian.learning_rules import OjasRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer
from pytorch_hebbian.visualizers import TensorBoardVisualizer


def main():
    # Creating the model
    model = models.create_fc1_model([28 ** 2, 16])

    # Creating the transforms, dataset and data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Creating the learning rule, optimizer and trainer
    learning_rule = OjasRule()
    optimizer = Local(named_params=model.named_parameters(), lr=0.01)
    visualizer = TensorBoardVisualizer(run='oja-test',
                                       log_dir=os.path.join(config.OUTPUT_DIR, 'tensorboard', 'runs.temp'))
    trainer = HebbianTrainer(model=model, learning_rule=learning_rule, optimizer=optimizer, visualizer=visualizer)

    # Running the trainer
    trainer.run(train_loader=train_loader, epochs=1, vis_weights_every=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=config.LOGGING_FORMAT)
    logging.getLogger("ignite").setLevel(logging.WARN)
    logging.getLogger("pytorch_hebbian").setLevel(logging.INFO)

    main()
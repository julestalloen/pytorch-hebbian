from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import models
from pytorch_hebbian import config
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer


def main():
    # Creating the model
    model = models.create_fc1_model([28 ** 2, 2000])

    # Creating the transforms, dataset and data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.MNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Creating the learning rule, optimizer and trainer
    learning_rule = KrotovsRule()
    optimizer = Local(named_params=model.named_parameters(), lr=0.01)
    trainer = HebbianTrainer(model=model, learning_rule=learning_rule, optimizer=optimizer)

    # Running the trainer
    trainer.run(train_loader=train_loader, epochs=10)


if __name__ == '__main__':
    main()

import logging

import torch
import torchvision
from torchvision import datasets, transforms

from pytorch_hebbian.utils.visualization import show_image
from pytorch_hebbian.learning_rules.krotov import KrotovsRule
from pytorch_hebbian.learning_engines.hebbian_engine import HebbianEngine
from pytorch_hebbian.optimizers.local import Local


def main_mnist():
    input_size, hidden_units, output_size = 784, 100, 100
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_units, bias=False),
        # torch.nn.ReLU(),
        # torch.nn.Linear(hidden_units, output_size),
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.mnist.FashionMNIST(root='../datasets', download=True, transform=transform)
    # dataset = datasets.mnist.MNIST(root='../datasets', download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    # Visualize some random images
    images, labels = next(iter(data_loader))
    show_image(torchvision.utils.make_grid(images[:64]), title='Some input samples')

    epochs = 100
    learning_rule = KrotovsRule(delta=0.2, k=2, norm=2)
    optimizer = Local(params=model.parameters(), lr=0.02)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs)
    learning_engine = HebbianEngine(learning_rule=learning_rule,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    visualize_weights=True)
    weights = learning_engine.train(model=model, data_loader=data_loader, epochs=epochs)

    print(weights[0].shape)
    # draw_weights(weights, dataset[0].shape, 40, 40)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(module)s:%(message)s')
    main_mnist()

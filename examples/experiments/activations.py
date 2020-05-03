import math
import os

import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
from pytorch_hebbian import config, utils

matplotlib.use('TkAgg')
PATH = os.path.dirname(os.path.abspath(__file__))
layer = torch.nn.Module
layer_outputs = torch.zeros(2000)
input_shape = (1, 28, 28)


def hook_fn(_, __, output):
    global layer_outputs
    layer_outputs = output


def plot_activations(weights, activations):
    global input_shape
    weights = weights.view(-1, *input_shape)
    weights = torch.gather(weights, 0, activations)
    # weights = weights[activations.bool(), :]
    print(weights.shape)
    num_weights = weights.shape[0]
    nrow = math.ceil(math.sqrt(num_weights))
    grid = torchvision.utils.make_grid(weights, nrow=nrow)

    fig = plt.figure()
    if weights.shape[1] == 1:
        grid_np = grid[0, :].cpu().numpy()
        nc = np.amax(np.absolute(grid_np))
        im = plt.imshow(grid_np, cmap='bwr', vmin=-nc, vmax=nc)
        plt.colorbar(im, ticks=[np.amin(grid_np), 0, np.amax(grid_np)])
    else:
        grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))
        grid_min = np.amin(grid_np)
        grid_max = np.amax(grid_np)
        grid_np = (grid_np - grid_min) / (grid_max - grid_min)
        plt.imshow(grid_np)
    plt.axis('off')
    fig.tight_layout()
    plt.show()


def visualize_activations(inputs):
    global layer_outputs
    for i in range(layer_outputs.shape[0]):
        inp = inputs[i, :].numpy()
        # TODO: WIP
        sort, indices = torch.sort(layer_outputs[i, :], 0)
        first_pos_index = (sort <= 0).sum(dim=0)
        activations = indices[first_pos_index:]
        # activations = (layer_outputs[i, :] > 0)
        nc = np.amax(np.absolute(inp))
        plt.imshow(np.transpose(inp, (1, 2, 0))[:, :, 0], cmap='bwr', vmin=-nc, vmax=nc)
        plt.show()
        plot_activations(layer.weight, activations)


def main():
    global layer
    model = models.create_fc1_model([28 ** 2, 2000], n=1, batch_norm=True)
    weights_path = "../../output/models/heb-mnist-fashion-20200426-101420_m_500_acc=0.852.pth"
    model = utils.load_weights(model, os.path.join(PATH, weights_path), layer_names=['linear1'], freeze=True)

    hooks = {}
    for name, p in model.named_children():
        if name == "repu":
            hooks[name] = p.register_forward_hook(hook_fn)
        elif name == "linear1":
            layer = p

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for (inputs, labels) in data_loader:
        model(inputs)
        visualize_activations(inputs)
        break


if __name__ == '__main__':
    main()

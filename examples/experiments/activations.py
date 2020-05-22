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
repu_outputs = torch.zeros(2000)
input_shape = (1, 28, 28)


def hook_fn(_, __, output):
    global repu_outputs
    repu_outputs = output


def plot_weights(weights, activation_indices):
    global input_shape

    weights = weights.view(-1, *input_shape)
    weights = weights[activation_indices, :]
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

    return weights


def plot_overlay(activated_weights, inp, activations):
    for i in range(activated_weights.shape[0]):
        unit = activated_weights[i, :].numpy()
        overlay = np.multiply(inp, unit)

        images = [inp, unit, overlay]
        ticks_min = np.amin(images)
        ticks_max = np.amax(images)
        nc = np.amax(np.absolute(images))
        print(ticks_min, ticks_max, nc)
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 3))

        im = None
        for j, ax in enumerate(axs):
            image = images[j]
            im = ax.imshow(np.transpose(image, (1, 2, 0))[:, :, 0], cmap='bwr', vmin=-nc, vmax=nc)

        fig.colorbar(im, ticks=[ticks_min, 0, ticks_max], ax=axs, shrink=0.7)
        fig.suptitle('Activation = {}'.format(activations[i]))
        # fig.tight_layout()
        plt.show()


def visualize_activations(inputs):
    global repu_outputs

    # Iterate over batch
    for i in range(repu_outputs.shape[0]):
        inp = inputs[i, :].numpy()

        sorted_outputs, sorted_indices = torch.sort(repu_outputs[i, :], 0, descending=True)
        cutoff = 0
        print("cutoff = {}".format(cutoff))
        first_neg_index = (sorted_outputs > cutoff).sum(dim=0)
        activation_indices = sorted_indices[:first_neg_index]
        activations = sorted_outputs[:first_neg_index]
        print("activations min: {}, max: {}".format(min(activations), max(activations)))
        print("{} activated neurons".format(len(activation_indices)))

        nc = np.amax(np.absolute(inp))
        im = plt.imshow(np.transpose(inp, (1, 2, 0))[:, :, 0], cmap='bwr', vmin=-nc, vmax=nc)
        plt.colorbar(im, ticks=[np.amin(inp), 0, np.amax(inp)])
        plt.show()

        activated_weights = plot_weights(layer.weight, activation_indices)
        plot_overlay(activated_weights, inp, activations)


def main():
    with torch.no_grad():
        global layer
        model = models.create_fc1_model([28 ** 2, 2000], n=1, batch_norm=True)
        weights_path = "../../output/models/heb-mnist-fashion-20200426-101420_m_500_acc=0.852.pth"
        model = utils.load_weights(model, os.path.join(PATH, weights_path), layer_names=['linear1', 'batch_norm'])

        for name, p in model.named_children():
            if name == "repu":
                p.register_forward_hook(hook_fn)
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

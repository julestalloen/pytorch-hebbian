import math

import numpy as np
import matplotlib.pyplot as plt

from pytorch_hebbian.visualizers.visualizer import Visualizer


class PerceptronVisualizer(Visualizer):

    def __init__(self):
        self.fig = plt.figure()

    def update(self, weights, shape, height=None, width=None):
        plt.pause(0.001)
        plt.clf()

        # If no height or width is specified create the smallest fitting square grid
        num_weights = weights.shape[0]
        if height is None or width is None:
            height = width = math.ceil(math.sqrt(num_weights))

        # Put all perceptrons in a single array
        dim_y, dim_x, depth = shape
        data = np.zeros((dim_y * height, dim_x * width, depth))

        yy = 0
        for y in range(height):
            for x in range(width):
                if yy < num_weights:
                    perceptron = weights[yy].reshape((depth, dim_y, dim_x)).transpose((1, 2, 0))
                    data[y * dim_y:(y + 1) * dim_y, x * dim_x:(x + 1) * dim_x, :] = perceptron
                    yy += 1
                else:
                    break

        # Depending on the amount of color channels, render the data
        if depth > 1:
            plt.imshow((data - np.amin(data)) / (np.amax(data) - np.amin(data)))
        else:
            nc = np.amax(np.absolute(data))
            im = plt.imshow(np.squeeze(data), cmap='bwr', vmin=-nc, vmax=nc)
            self.fig.colorbar(im, ticks=[np.amin(data), 0, np.amax(data)])

        plt.axis('off')
        self.fig.canvas.draw()

    def __del__(self):
        plt.close()

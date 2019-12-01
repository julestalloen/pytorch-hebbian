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

        # TODO: support grids with non-square sizes
        if height is None or width is None:
            height = int(math.sqrt(weights.shape[0]))
            width = height

        dim_y, dim_x, depth = shape
        data = np.zeros((dim_y * height, dim_x * width, depth))

        yy = 0
        for y in range(height):
            for x in range(width):
                perceptron = weights[yy].reshape((depth, dim_y, dim_x)).transpose((1, 2, 0))
                data[y * dim_y:(y + 1) * dim_y, x * dim_x:(x + 1) * dim_x, :] = perceptron
                yy += 1

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

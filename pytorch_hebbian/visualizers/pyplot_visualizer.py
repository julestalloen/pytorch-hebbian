import matplotlib.pyplot as plt
import numpy as np

from pytorch_hebbian.utils.visualization import weights_to_grid
from pytorch_hebbian.visualizers.weights_visualizer import WeightsVisualizer


class PyplotVisualizer(WeightsVisualizer):

    def __init__(self):
        self.fig = plt.figure()

    def update(self, weights, shape, step):
        plt.pause(0.001)
        plt.clf()

        data = weights_to_grid(weights, shape)

        # Depending on the amount of color channels, render the data
        if shape[2] > 1:
            plt.imshow((data - np.amin(data)) / (np.amax(data) - np.amin(data)))
        else:
            nc = np.amax(np.absolute(data))
            im = plt.imshow(np.squeeze(data), cmap='bwr', vmin=-nc, vmax=nc)
            self.fig.colorbar(im, ticks=[np.amin(data), 0, np.amax(data)])

        plt.axis('off')
        plt.title('Step: {}'.format(step))
        self.fig.canvas.draw()

    def __del__(self):
        plt.close()

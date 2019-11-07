import logging
import os

import scipy.io
import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))


def load(filename='mnist_all.mat'):
    mat = scipy.io.loadmat(os.path.join(PATH, filename))

    classes = 10
    dim = 28
    size = dim**2
    data = np.zeros((0, size))

    for i in range(classes):
        data = np.concatenate((data, mat['train' + str(i)]), axis=0)

    data /= 255.0
    data = np.array([np.reshape(n, (dim, dim)) for n in data])

    logging.info('Returning numpy array with shape {}'.format(data.shape))
    return data

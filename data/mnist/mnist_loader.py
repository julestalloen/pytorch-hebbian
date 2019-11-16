import os
import gzip
import logging

import numpy as np
import scipy.io

from data.data_loader import ImageLoader

PATH = os.path.dirname(os.path.abspath(__file__))


class MnistLoader(ImageLoader):

    def __init__(self, dataset: str = 'fashion'):
        self.dataset = dataset
        self.path = os.path.join(PATH, 'data')

    def load_digits(self):
        filename = 'mnist_all.mat'
        mat = scipy.io.loadmat(os.path.join(self.path, filename))

        classes = 10
        dim = 28
        size = dim ** 2
        data = np.zeros((0, size))

        for i in range(classes):
            data = np.concatenate((data, mat['train' + str(i)]), axis=0)

        data = np.array([np.reshape(n, (dim, dim)) for n in data])

        return data

    def load_fashion(self):
        labels_path = os.path.join(self.path, 'train-labels-idx1-ubyte.gz')
        images_path = os.path.join(self.path, 'train-images-idx3-ubyte.gz')

        with gzip.open(labels_path, 'rb') as lb_path:
            labels = np.frombuffer(lb_path.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as img_path:
            images = np.frombuffer(img_path.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

        return images

    def _load(self):
        if self.dataset == 'digits':
            return self.load_digits()
        elif self.dataset == 'fashion':
            return self.load_fashion()
        else:
            logging.warning('Invalid dataset')

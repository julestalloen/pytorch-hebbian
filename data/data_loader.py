from abc import ABC, abstractmethod
import logging
import os

import numpy as np
import cv2 as cv

import config
from utils.image import resize_image


class DataLoader(ABC):

    def load(self):
        return self._load()

    @abstractmethod
    def _load(self) -> list:
        pass


class ImageLoader(DataLoader):

    def load(self, samples: int = None, gray: bool = False, max_resolution: tuple = None,
             cache: bool = False) -> np.array:
        data = np.array(self._load())

        # Limit the number of returned samples
        if samples:
            np.random.shuffle(data)
            data = data[:samples]

        # Enforce max resolution
        if max_resolution:
            data = [resize_image(image, max_height=max_resolution[0], max_width=max_resolution[1]) for image in data]

        # Grayscale conversion
        if gray:
            data = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in data]
            logging.info('Converted images to grayscale.')

        # Normalization
        data = np.divide(data, 255)

        # Caching
        if cache:
            np.save('temp.npy', data)
            logging.info('Cached numpy array with shape {}.'.format(data.shape))

        logging.info('Returning numpy array with shape {}.'.format(data.shape))

        return data

    @abstractmethod
    def _load(self) -> list:
        pass

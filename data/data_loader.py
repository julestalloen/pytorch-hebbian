from abc import ABC, abstractmethod
import logging

import numpy as np
import cv2 as cv


class DataLoader(ABC):

    def load(self, gray: bool = False, samples: int = None) -> np.array:
        data = np.array(self._load())

        # Grayscale conversion
        if gray:
            data = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in data]

        # Limit the number of returned samples
        if samples:
            np.random.shuffle(data)
            data = data[:samples]

        # Normalization
        data = np.divide(data, 255)

        logging.info('Returning numpy array with shape {}'.format(data.shape))

        return data

    @abstractmethod
    def _load(self) -> list:
        pass

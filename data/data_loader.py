from abc import ABC, abstractmethod
import logging

import numpy as np


class DataLoader(ABC):

    def load(self):
        return self._load()

    @abstractmethod
    def _load(self) -> list:
        pass


class ImageLoader(DataLoader):

    def load(self, samples: int = None, shuffle: bool = True) -> np.array:
        data = np.array(self._load())

        # Limit the number of returned samples
        if samples:
            np.random.shuffle(data)
            data = data[:samples]

        if shuffle:
            np.random.shuffle(data)

        logging.info('Loaded numpy array with shape {}.'.format(data.shape))

        return data

    @abstractmethod
    def _load(self) -> list:
        pass

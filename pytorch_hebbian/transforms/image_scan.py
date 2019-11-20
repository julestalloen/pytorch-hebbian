import logging
from math import floor

from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np


class ImageScan:

    def __init__(self, images):
        self.images = images

    def scan(self, shape: tuple = (32, 32)) -> np.array:
        all_patches = []

        for image in self.images:
            h, w = image.shape
            max_patches = floor(h / shape[0]) * floor(w / shape[1])

            logging.debug('Max patches = {}'.format(max_patches))

            patches = extract_patches_2d(image, shape, max_patches=max_patches)
            all_patches.extend(patches)

        return np.array(all_patches)

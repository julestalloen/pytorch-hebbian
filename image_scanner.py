from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np


class ImageScanner:

    def __init__(self, images):
        self.images = images

    def scan(self, shape: tuple = (32, 32)) -> np.array:
        all_patches = []

        for image in self.images:
            print('Image shape: {}'.format(image.shape))

            patches = extract_patches_2d(image, shape, max_patches=256)

            print('Patches shape: {}'.format(patches.shape))

            all_patches.extend(patches)

        return np.array(all_patches)

import numpy as np
import cv2 as cv

from preprocessing.preprocessor import ImagePreprocessor
from utils.image import resize_image


class CommonPreprocessor(ImagePreprocessor):

    def __init__(self, rescale: float = None, max_res: tuple = None, gray: bool = False):
        self.rescale = rescale
        self.max_res = max_res
        self.gray = gray

    def process_image(self, image):
        # Enforce max resolution
        if self.max_res is not None:
            image = resize_image(image, max_height=self.max_res[0], max_width=self.max_res[1])

        # Grayscale conversion
        if self.gray:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Rescale
        image = np.multiply(image, self.rescale)

        return image

    def process(self, images):
        return np.apply_along_axis(self.process_image, 0, images)

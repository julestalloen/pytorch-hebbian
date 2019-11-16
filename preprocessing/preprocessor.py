from abc import ABC, abstractmethod


class ImagePreprocessor(ABC):

    @abstractmethod
    def process_image(self, image):
        pass

    @abstractmethod
    def process(self, images):
        pass

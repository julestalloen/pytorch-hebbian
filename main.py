import logging

import numpy as np

from data.mnist.mnist_loader import MnistLoader
# from data.coco.coco_loader import CocoLoader
from preprocessing.common_preprocessor import CommonPreprocessor
from utils.visualization import draw_weights, show_image_grid
from learning_engines.hebbian_engine import HebbianEngine
from learning_rules.krotov import KrotovsRule
from optimizers.linear import Linear
from image_scanner import ImageScanner


def main():
    # data_loader = CocoLoader(categories=['dog'])
    # data = data_loader.load(gray=True)
    # np.save('data.npy', data)

    data = np.load('data/coco/cache/dog.npy', allow_pickle=True)

    image_scanner = ImageScanner(data)
    data = image_scanner.scan(shape=(9, 9))

    show_image_grid(data)

    learning_rule = KrotovsRule(delta=0.1, k=3, norm=3)
    optimizer = Linear(learning_rate=0.05)
    learning_engine = HebbianEngine(learning_rule=learning_rule, optimizer=optimizer, visualize_weights=True)
    weights = learning_engine.fit(400, data, epochs=500, batch_size=100)

    draw_weights(weights, data[0].shape, 20, 20)


def main_mnist():
    data_loader = MnistLoader(dataset='fashion')
    data = data_loader.load()

    image_scanner = ImageScanner(data)
    data = image_scanner.scan(shape=(8, 8))
    show_image_grid(data)

    preprocessor = CommonPreprocessor(rescale=1/255)
    data = preprocessor.process(data)

    learning_rule = KrotovsRule(delta=0.1, k=3, norm=2)
    optimizer = Linear(learning_rate=0.02)
    learning_engine = HebbianEngine(learning_rule=learning_rule, optimizer=optimizer, visualize_weights=True)
    weights = learning_engine.fit(1600, data, epochs=100, batch_size=100)

    draw_weights(weights, data[0].shape, 40, 40)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main_mnist()

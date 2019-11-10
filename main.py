import logging

# import numpy as np
# import cv2 as cv

# from data.mnist import mnist
# from data.cifar import cifar
from data.coco.coco_loader import CocoLoader
from util.visualization import draw_weights, show_image_patches
from learning_engines.hebbian_engine import HebbianEngine
# from learning_rules.hebb import HebbsRule
# from learning_rules.oja import OjasRule
from learning_rules.krotov import KrotovsRule
from image_scanner import ImageScanner


def main():
    data_loader = CocoLoader()
    data = data_loader.load(gray=True)
    image_scanner = ImageScanner(data)
    data = image_scanner.scan()

    show_image_patches(data[:81])

    learning_rule = KrotovsRule(delta=0.3, k=3, norm=3)
    learning_engine = HebbianEngine(learning_rule=learning_rule, learning_rate=0.02, visualize_weights=True)
    synapses = learning_engine.fit(100, data, epochs=5000, batch_size=100)

    draw_weights(synapses, data[0].shape, 10, 10)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

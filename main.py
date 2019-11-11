import logging

# from matplotlib import pyplot as plt

# from data.mnist import mnist
# from data.cifar import cifar
from data.coco.coco_loader import CocoLoader
from utils.visualization import draw_weights, show_image_patches
from learning_engines.hebbian_engine import HebbianEngine
# from learning_rules.hebb import HebbsRule
# from learning_rules.oja import OjasRule
from learning_rules.krotov import KrotovsRule
from optimizers.linear import Linear
from image_scanner import ImageScanner


def main():
    data_loader = CocoLoader(categories=['dog'])
    data = data_loader.load(gray=True)
    image_scanner = ImageScanner(data)
    data = image_scanner.scan(shape=(16, 16))

    show_image_patches(data[:81])

    learning_rule = KrotovsRule(delta=0.3, k=2, norm=2)
    optimizer = Linear(learning_rate=0.04)
    learning_engine = HebbianEngine(learning_rule=learning_rule, optimizer=optimizer, visualize_weights=True)
    weights = learning_engine.fit(100, data, epochs=100, batch_size=100)

    draw_weights(weights, data[0].shape, 10, 10)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

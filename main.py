import logging

import numpy as np
# import cv2 as cv

# from data.mnist import mnist
from data.cifar import cifar
from util.visualization import draw_weights
from learning_engines.hebbian_engine import HebbianEngine
# from learning_rules.hebb import HebbsRule
from learning_rules.oja import OjasRule
# from learning_rules.krotov import KrotovsRule


def main():
    data = cifar.load_gray()
    np.random.shuffle(data)
    data = data[:500]

    # learning_rule = KrotovsRule(delta=0.01, k=3, norm=4)
    learning_rule = OjasRule(c=0.01)
    learning_engine = HebbianEngine(learning_rule=learning_rule, learning_rate=0.002, visualize_weights=True)
    synapses = learning_engine.fit(100, data, epochs=1, batch_size=1)

    draw_weights(synapses, data[0].shape, 10, 10)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()

"""
https://www.cs.toronto.edu/~kriz/cifar.html
"""
import os
import pickle

import numpy as np
import cv2 as cv

PATH = os.path.dirname(os.path.abspath(__file__))


def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def load(dir_name='cifar-10-batches-py'):
    dir_path = os.path.join(PATH, dir_name)
    data = []

    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)

        if file.startswith('data') or file.startswith('test'):
            dic = unpickle(file_path)
            data.extend(dic[b'data'])

    data = [np.transpose(np.reshape(array, (3, 32, 32)), (1, 2, 0)) for array in data]
    data = np.array(data)

    return np.array(data)


def load_gray(dir_name='cifar-10-batches-py'):
    color_data = load(dir_name)
    data = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in color_data]

    data = np.divide(data, 255)

    return np.array(data)

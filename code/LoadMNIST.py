import numpy as np
from matplotlib import pyplot as plt

base_path = 'C:/Meghna/ASU/S1/FSL/project/data/'


def load_fashion_mnist():
    fd = open(base_path + 'train-images-idx3-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(base_path + 'train-labels-idx1-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(base_path + 't10k-images-idx3-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(base_path + 't10k-labels-idx1-ubyte')
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData / 255.
    tsData = tsData / 255.

    return trData, trLabels, tsData, tsLabels

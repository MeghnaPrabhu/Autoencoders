import numpy as np


class GeneralUtils:

    @staticmethod
    def load_fashion_mnist(base_path):
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

    @staticmethod
    def get_label_dict():
        label_dict = {0: 'T-shirt_top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                      6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle_boot'
                      }
        return label_dict

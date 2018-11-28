import numpy as np
import copy
from network import multi_layer_network


class DenoisingAutoencoder:

    def __init__(self, train_data, train_label, validation_data, validation_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.validation_data = validation_data
        self.validation_label = validation_label
        self.test_data = test_data
        self.test_label = test_label

    # TODO: need to refactor, very slow for 60k images. Check if faster implementation exists.
    def add_noise(self, input_data):
        # noise_level is the number of pixels that are randomly chosen and corrupted by setting their values to 0
        noise_level = 400
        # setting 400 random pixels to 0 in every image
        corrupted_images = []
        for i in range(input_data.shape[1]):
            corrupted_pixels = np.random.randint(0, input_data.shape[0], noise_level)
            corrupted_image = copy.copy(input_data[:, i])
            for pixel in corrupted_pixels:
                corrupted_image[pixel] = 0
            corrupted_images.append(corrupted_image)
        return np.array(corrupted_images).T

    def train(self):
        net_dims = [784, 1024, 784]
        num_iterations = 500
        learning_rate = 0.1
        decay_rate = 0.01
        corrupted_images = self.add_noise(self.train_data)
        costs, validation_costs, parameters = \
            multi_layer_network(self.train_data, self.train_label, self.validation_data, self.validation_label,
                                net_dims, corrupted_input=corrupted_images, num_iterations=num_iterations,
                                learning_rate=learning_rate, decay_rate=decay_rate)

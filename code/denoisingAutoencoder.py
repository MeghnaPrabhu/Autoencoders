import numpy as np
import copy
from network import multi_layer_network
from network import multi_layer_forward
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


class DenoisingAutoencoder:

    def __init__(self, base_path, train_data, train_label, validation_data, validation_label, test_data, test_label):
        self.base_path = base_path
        self.train_data = train_data
        self.train_label = train_label
        self.validation_data = validation_data
        self.validation_label = validation_label
        self.test_data = test_data
        self.test_label = test_label

    # TODO: need to refactor, very slow for 60k images. Check if faster implementation exists.
    def add_noise(self, input_data, noise_level):
        # noise_level is the number of pixels that are randomly chosen and corrupted by setting their values to 0
        num_pixels_to_set_to_zero = int(input_data.shape[0] * noise_level / 100)
        # setting 400 random pixels to 0 in every image
        corrupted_images = []
        for i in range(input_data.shape[1]):
            corrupted_pixels = np.random.randint(0, input_data.shape[0], num_pixels_to_set_to_zero)
            corrupted_image = copy.copy(input_data[:, i])
            for pixel in corrupted_pixels:
                corrupted_image[pixel] = 0
            corrupted_images.append(corrupted_image)
        return np.array(corrupted_images).T

    def train(self):
        net_dims = [784, 1024, 784]
        num_iterations = 1000
        learning_rate = 0.1
        decay_rate = 0.01
        print('Enter % of noise you want to add: ')
        noise_level = int(input())
        corrupted_images = self.add_noise(self.train_data, noise_level)
        network_type = 'DAE'
        f_activation = 'sigmoid'
        h_activation = 'relu'
        parameters_path = self.base_path + "/" + "parameters" + str(learning_rate).replace(".", "_") + str(
            net_dims[0]) + str(
            net_dims[1]) + str(net_dims[2]) + str(f_activation) + str(h_activation) + str(num_iterations) + str(noise_level)
        try:
            parameters = pickle.load(open(parameters_path + ".pickle", "rb"))
        except (OSError, IOError) as e:

            costs, parameters = \
                multi_layer_network(self.train_data, self.train_label, self.validation_data, self.validation_label,
                                    net_dims, network_type, corrupted_input=corrupted_images,
                                    num_iterations=num_iterations,
                                    learning_rate=learning_rate, decay_rate=decay_rate, activation_h=h_activation, activation_f= f_activation)
            pickle.dump(parameters, open(parameters_path + ".pickle", "wb"))

            # Ploting cost function
            plt.plot(costs, label='training data');

            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("learning rate =" + str(learning_rate))
            plt.legend(loc='upper right')
            plt.show()

        '''For verification'''
        #parameters['W2'] = parameters['W1'].T
        AL, cache = multi_layer_forward(corrupted_images, parameters, h_activation, f_activation)
        for i in range(0, self.train_data.shape[1], int(self.train_data.shape[1] / 10)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle('Denoising Autoencoder', fontsize=20)
            fig.set_size_inches(18.5, 10.5, forward=True)
            ax1.imshow(np.reshape(self.train_data.T[i], (28, 28)))
            ax1.set_title("Actual Input")

            ax2.imshow(np.reshape(corrupted_images.T[i], (28, 28)))
            ax2.set_title("Noised input")

            ax3.imshow(np.reshape(AL.T[i], (28, 28)))
            ax3.set_title("De-Noised output")

            plt.show()


        # parameters['W1'] = parameters['W2'].T
        # AL, cache = multi_layer_forward(corrupted_images, parameters)
        # for i in range(0, self.train_data.shape[1], int(self.train_data.shape[1] / 10)):
        #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #     fig.suptitle('Denoising Autoencoder W2', fontsize=20)
        #     fig.set_size_inches(18.5, 10.5, forward=True)
        #     ax1.imshow(np.reshape(self.train_data.T[i], (28, 28)))
        #     ax1.set_title("Actual Input")
        #
        #     ax2.imshow(np.reshape(corrupted_images.T[i], (28, 28)))
        #     ax2.set_title("Noised input")
        #
        #     ax3.imshow(np.reshape(AL.T[i], (28, 28)))
        #     ax3.set_title("De-Noised output")
        #
        #     plt.show()

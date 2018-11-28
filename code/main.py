import numpy as np
from LoadMNIST import load_fashion_mnist
from denoisingAutoencoder import DenoisingAutoencoder
from stackedAutoencoders import StackedAutoencoder


def split_training_data(train_data_initial, train_label_initial):
    train_data = train_data_initial[:, :4000]
    validation_data = train_data_initial[:, 4000:5000]
    train_label = train_label_initial[:, :4000]
    validation_label = train_label_initial[:, 4000:5000]
    for i in range(5000, 60000, 6000):
        train_data = np.concatenate((train_data, train_data_initial[:, i:i + 4000]), axis=1)
        validation_data = np.concatenate((validation_data, train_data_initial[:, i + 4000:i + 5000]), axis=1)
        train_label = np.concatenate((train_label, train_label_initial[:, i:i + 4000]), axis=1)
        validation_label = np.concatenate((validation_label, train_label_initial[:, i + 4000:i + 5000]), axis=1)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


if __name__ == "__main__":
    train_data_initial, train_label_initial, test_data, test_label = load_fashion_mnist(noTrSamples=5000,
                                                                                        noTsSamples=1000,
                                                                                        digit_range=[0, 1, 2, 3, 4, 5,
                                                                                                     6, 7, 8, 9],
                                                                                        noTrPerClass=500,
                                                                                        noTsPerClass=100)
    train_data, train_label, validation_data, validation_label, test_data, test_label = \
        split_training_data(train_data_initial, train_label_initial)
    keep_running = True
    while keep_running:
        print("Select network 1. Denoising Autoencoder 2. Stacked Autoencoder: ")
        network_option = int(input())
        if network_option == 1:
            DenoisingAutoencoder(train_data, train_label, validation_data,
                                 validation_label, test_data, test_label).train()
        elif network_option == 2:
            stacked_AE =StackedAutoencoder(train_data, train_label, validation_data,
                               validation_label, test_data, test_label)
            stacked_AE.train()
        print("Press N to stop")
        input_option = input()
        if input_option == "N":
            keep_running = False

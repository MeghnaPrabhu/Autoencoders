import numpy as np
from LoadMNIST import load_fashion_mnist
from denoisingAutoencoder import DenoisingAutoencoder
from stackedAutoencoders import StackedAutoencoder
import sys


def split_training_data(train_data_initial, train_label_initial, noTrSamples, noTrPerClass):
    # TODO: uncomment and correct this
    # train_data = train_data_initial[:, :1000]
    # validation_data = train_data_initial[:, 1000:1200]
    # train_label = train_label_initial[:, :1000]
    # validation_label = train_label_initial[:, 1000:1200]

    train_data = train_data_initial
    validation_data = train_data_initial
    train_label = train_label_initial
    validation_label = train_label_initial
    # for i in range(5000, 60000, 6000):
    #     train_data = np.concatenate((train_data, train_data_initial[:, i:i + 4000]), axis=1)
    #     validation_data = np.concatenate((validation_data, train_data_initial[:, i + 4000:i + 5000]), axis=1)
    #     train_label = np.concatenate((train_label, train_label_initial[:, i:i + 4000]), axis=1)
    #     validation_label = np.concatenate((validation_label, train_label_initial[:, i + 4000:i + 5000]), axis=1)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


if __name__ == "__main__":
    base_path = sys.argv[1]
    print("Enter number of training data:")
    train_count = int(input())
    noTrSamples = train_count
    noTrPerClass = noTrSamples / 10
    train_data_initial, train_label_initial, test_data, test_label = load_fashion_mnist(base_path,
                                                                                        noTrSamples=noTrSamples,
                                                                                        noTsSamples=1000,
                                                                                        digit_range=[0, 1, 2, 3, 4, 5,
                                                                                                     6, 7, 8, 9],
                                                                                        noTrPerClass=int(noTrPerClass),
                                                                                        noTsPerClass=100)

    train_data, train_label, validation_data, validation_label, test_data, test_label = \
        split_training_data(train_data_initial, train_label_initial, noTrSamples, noTrPerClass)

    keep_running = True
    while keep_running:
        print(
            "Select network 1. Denoising Autoencoder 2. Stacked Autoencoder 3.KNN 4. SVM classifier after "
            "Autoencoders(Deep) reduction 5. SVM after PCA 6. SVM on full data")
        network_option = int(input())

        if network_option == 1:
            DenoisingAutoencoder(base_path, train_data, train_label, validation_data,
                                 validation_label, test_data, test_label).train()
        elif network_option >= 2:
            stacked_AE = StackedAutoencoder(base_path, train_data, train_label, validation_data,
                                            validation_label, test_data, test_label)

            stacked_AE.train(network_option)
        print("Press N to stop")
        input_option = input()
        if input_option == "N":
            keep_running = False

from LoadMNIST import load_fashion_mnist
from denoisingAutoencoder import DenoisingAutoencoder
from stackedAutoencoders import StackedAutoencoder


def split_training_data(train_data_initial, train_label_initial):
    # TODO: fill here
    train_data = train_data_initial
    train_label = train_label_initial
    test_data = train_data_initial
    test_label = train_label
    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
    train_data_initial, train_label_initial, test_data, test_label = load_fashion_mnist()
    # TODO: split data as train and validation
    train_data, train_label, test_data, test_label = split_training_data(train_data_initial, train_label_initial)
    keep_running = True
    while keep_running:
        print("Select network 1. Denoising Autoencoder 2. Stacked Autoencoder: ")
        network_option = int(input())

        if network_option == 1:
            DenoisingAutoencoder(train_data_initial, train_label_initial, test_data, test_label)
        elif network_option == 2:
            StackedAutoencoder(train_data_initial, train_label_initial, test_data, test_label)

        print("Press N to stop")
        input_option = input()
        if input_option == "N":
            keep_running = False

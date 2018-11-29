import numpy as np
from LoadMNIST import load_fashion_mnist
from denoisingAutoencoder import DenoisingAutoencoder
from stackedAutoencoders import StackedAutoencoder


# def split_training_data(train_data_initial, train_label_initial):
#     train_data = train_data_initial[:, :4000]
#     validation_data = train_data_initial[:, 4000:5000]
#     train_label = train_label_initial[:, :4000]
#     validation_label = train_label_initial[:, 4000:5000]
#     for i in range(5000, 60000, 6000):
#         train_data = np.concatenate((train_data, train_data_initial[:, i:i + 4000]), axis=1)
#         validation_data = np.concatenate((validation_data, train_data_initial[:, i + 4000:i + 5000]), axis=1)
#         train_label = np.concatenate((train_label, train_label_initial[:, i:i + 4000]), axis=1)
#         validation_label = np.concatenate((validation_label, train_label_initial[:, i + 4000:i + 5000]), axis=1)
#
#     return train_data, train_label, validation_data, validation_label, test_data, test_label

def split_training_data(X,Y):
    X_val, Y_val, X_train, Y_train = None, None, None, None
    batch_size = 1000

    for i in range(10):
        X_temp, Y_temp = X[:, i*600:(i+1)*600], Y[:, i*600:(i+1)*600]

        if X_train is None:
            X_train, Y_train = X_temp[:, :500], Y_temp[:, :500]
            X_val, Y_val = X_temp[:, 500:600], Y_temp[:, 500:600]
        else:
            X_train, Y_train = np.concatenate((X_train, X_temp[:, :500]), axis=1), np.concatenate((Y_train, Y_temp[:, :500]), axis=1)
            X_val, Y_val = np.concatenate((X_val, X_temp[:, 500:600]), axis=1), np.concatenate((Y_val, Y_temp[:, 500:600]), axis=1)

    batches_x = np.split(X,batch_size,axis=1)
    batches_y = np.split(Y,batch_size,axis=1)
    n_batches = len(batches_x)

    return X_train,Y_train,X_val,Y_val,batches_x,batches_y,n_batches



if __name__ == "__main__":
    train_data_initial, train_label_initial, test_data, test_label = load_fashion_mnist(noTrSamples=5000,
                                                                                        noTsSamples=1000,
                                                                                        digit_range=[0, 1, 2, 3, 4, 5,
                                                                                                     6, 7, 8, 9],
                                                                                        noTrPerClass=500,
                                                                                        noTsPerClass=100)
    train_data, train_label, validation_data, validation_label, batches_x, batches_y, n_batches = \
    split_training_data(train_data_initial, train_label_initial)
    keep_running = True
    while keep_running:
        print("Select network 1. Denoising Autoencoder 2. Stacked Autoencoder: ")
        network_option = 1
        if network_option == 1:
            DenoisingAutoencoder(train_data, train_label, validation_data,
                                 validation_label, test_data, test_label).train(batches_x, batches_y, n_batches)
        elif network_option == 2:
            stacked_AE =StackedAutoencoder(train_data, train_label, validation_data,
                               validation_label, test_data, test_label)
            stacked_AE.train()
        print("Press N to stop")
        input_option = input()
        if input_option == "N":
            keep_running = False

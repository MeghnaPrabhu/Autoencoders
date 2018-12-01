import numpy as np
from network import multi_layer_network, multi_layer_forward
import activations as Activation
from LoadMNIST import load_fashion_mnist


class StackedAutoencoder:

    def __init__(self, base_path, train_data, train_label, validation_data, validation_label, test_data, test_label):
        self.base_path = base_path
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.validation_data = validation_data
        self.validation_label = validation_label
        self.net_dims = [784, 500, 300, 100, 10]
        self.num_iterations = 1000
        self.learning_rate = 0.1

    def classify(self, X, parameters):

        AL, A = self.multi_layer_forward(X, parameters)
        ## call to softmax cross entropy loss
        YPred = np.exp(AL - np.max(AL, axis=0)) / np.sum(np.exp(AL - np.max(AL, axis=0)), axis=0, keepdims=True)
        YPred = np.argmax(YPred, axis=0)

        AL, A = self.multi_layer_forward(X, parameters)
        Ypred = np.exp(AL - np.max(AL, axis=0)) / np.sum(np.exp(AL - np.max(AL, axis=0)), axis=0, keepdims=True)
        Ypred = Ypred.argmax(axis=0)

        return YPred

    def linear(self, Z):
        '''
        computes linear activation of Z
        This function is implemented for completeness

        Inputs:
            Z is a numpy.ndarray (n, m)

        Returns:
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''
        A = Z
        return A

    def linear_der(self, dA):
        '''
        computes derivative of linear activation
        This function is implemented for completeness

        Inputs:
            dA is the derivative from subsequent layer. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}, where Z was the input
            to the activation layer during forward propagation

        Returns:
            dZ is the derivative. numpy.ndarray (n,m)
        '''
        dZ = np.array(dA, copy=True)
        return dZ

    def relu(self, Z):
        '''
        computes relu activation of Z

        Inputs:
            Z is a numpy.ndarray (n, m)

        Returns:
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''
        A = np.maximum(0, Z)
        return A

    def multi_layer_forward(self, X, parameters):
        '''
        Forward propgation through the layers of the network

        Inputs:
            X - numpy.ndarray (n,m) with n features and m samples
            parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
        Returns:
            AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
                where c is number of categories and m is number of samples in the batch
            caches - a dictionary of associated caches of parameters and network inputs
        '''
        L = len(parameters) // 2
        A = X
        for l in range(1, L):  # since there is no W0 and b0
            A = self.layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")

        AL = self.layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
        return AL, A

    def linear_forward(self, A, W, b):
        '''
        Input A propagates through the layer
        Z = WA + b is the output of this layer.

        Inputs:
            A - numpy.ndarray (n,m) the input to the layer
            W - numpy.ndarray (n_out, n) the weights of the layer
            b - numpy.ndarray (n_out, 1) the bias of the layer

        Returns:
            Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
            cache - a dictionary containing the inputs A
        '''
        ### CODE HERE
        Z = np.dot(W, A) + b
        return Z

    def layer_forward(self, A_prev, W, b, activation):
        '''
        Input A_prev propagates through the layer and the activation

        Inputs:
            A_prev - numpy.ndarray (n,m) the input to the layer
            W - numpy.ndarray (n_out, n) the weights of the layer
            b - numpy.ndarray (n_out, 1) the bias of the layer
            activation - is the string that specifies the activation function

        Returns:
            A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
            g is the activation function
            cache - a dictionary containing the cache from the linear and the nonlinear propagation
            to be used for derivative
        '''
        Z = self.linear_forward(A_prev, W, b)
        if activation == "relu":
            A = self.relu(Z)
        elif activation == "linear":
            A = self.linear(Z)

        return A

    def initialize_multilayer_weights(self):
        '''
        Initializes the weights of the multilayer network

        Inputs:
            net_dims - tuple of network dimensions

        Returns:
            dictionary of parameters
        '''
        np.random.seed(0)
        num_layers = len(self.net_dims)
        update_parameters = {}
        train_data = self.train_data

        for l in range(num_layers - 1):
            curr_layer_dims = [self.net_dims[l], self.net_dims[l + 1], self.net_dims[l]]
            # parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * 0.01
            # parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1) * 0.01
            # TODO: get weights from denoiser

            parameters_path = self.base_path + "/" + "parameters" + str(self.learning_rate).replace(".", "_") + str(
                self.net_dims[1]) + str(
                self.net_dims[2]) + str(self.net_dims[3]) + "sigmoid" + "relu" + str(self.num_iterations) + str(
                self.train_data.shape[1]) + str(l)
            costs_path = self.base_path + "/" + "costs" + str(self.learning_rate).replace(".", "_") + str(
                self.net_dims[1]) + str(
                self.net_dims[2]) + str(self.net_dims[3]) + "sigmoid" + "relu" + str(self.num_iterations) + str(
                self.train_data.shape[1]) + str(l)

            costs, parameters = multi_layer_network(train_data, self.train_label, None, None, curr_layer_dims,
                                                    "SAE",parameters_path,costs_path, None, 500, learning_rate=0.1, activation_h='relu',
                                                    activation_f='sigmoid')
            forward_param = {}
            update_parameters["W" + str(l + 1)] = parameters["W1"]
            update_parameters["b" + str(l + 1)] = parameters["b1"]
            forward_param["W1"] = parameters["W1"]
            forward_param["b1"] = parameters["b1"]

            train_data, _ = self.multi_layer_forward(train_data, parameters)

        update_parameters["W" + str(l + 1)] = np.random.randn(self.net_dims[num_layers - 1],
                                                              self.net_dims[num_layers - 2]) * np.sqrt(
            2.0 / self.net_dims[num_layers - 1])
        update_parameters["b" + str(l + 1)] = np.random.randn(self.net_dims[num_layers - 1], 1) * np.sqrt(
            2.0 / self.net_dims[num_layers - 2])

        return update_parameters

    def train(self):

        train_data_initial, train_label_initial, test_data, test_label = load_fashion_mnist(self.base_path,
                                                                                            noTrSamples=int(50),
                                                                                            noTsSamples=1000,
                                                                                            digit_range=[0, 1, 2, 3, 4,
                                                                                                         5,
                                                                                                         6, 7, 8, 9],
                                                                                            noTrPerClass=int(
                                                                                                5),
                                                                                            noTsPerClass=100)

        self.num_iterations = 1000
        parameters = self.initialize_multilayer_weights()
        # A = self.train_data
        self.learning_rate = 0.1
        for ii in range(self.num_iterations):
            self.train_data = train_data_initial
            self.train_label = train_label_initial
            A = self.train_data
            AL, A_prev = self.multi_layer_forward(A, parameters)
            A = np.exp(AL - np.max(AL, axis=0)) / np.sum(np.exp(AL - np.max(AL, axis=0)), axis=0, keepdims=True)
            loss = 0
            for i in range(0, self.train_label.shape[1]):
                index = int(self.train_label[0][i])
                loss = loss + np.log(A[index][i])
            loss = -loss / self.train_label.shape[1]
            one_hot_vector = np.zeros((self.train_label.size, int(self.train_label.max()) + 1))
            one_hot_vector[np.arange(self.train_label.size), self.train_label.astype(int)] = 1
            one_hot_vector = one_hot_vector.T
            dZ = A - one_hot_vector
            dZ = dZ / A.shape[1]
            dW = np.dot(dZ, A_prev.T)
            db = np.sum(dZ, axis=1, keepdims=True)

            parameters["W" + str(len(self.net_dims) - 1)] = parameters[
                                                                "W" + str(len(self.net_dims) - 1)] - (
                                                                    self.learning_rate * dW)
            parameters["b" + str(len(self.net_dims) - 1)] = parameters["b" + str(len(self.net_dims) - 1)] - (
                    self.learning_rate * db)

            print("Iteration ", ii, " : ", loss)

        # classify
        test_YPred = self.classify(self.test_data, parameters)
        tsAcc = ((test_YPred == self.test_label[0]).sum()) / (len(test_YPred) * 100)
        test_data_error = 0
        for index, pred in enumerate(test_YPred):
            if (self.test_label[0][index] != pred):
                test_data_error = test_data_error + 1

        teAcc = ((self.test_data.shape[1] - test_data_error) / self.test_data.shape[1]) * 100

        print('Accuracy test data: ', tsAcc)

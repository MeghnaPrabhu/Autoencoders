import numpy as np
import denoisingAutoencoder

class StackedAutoencoder:

    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

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
            A, cache = self.layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")

        AL, cache = self.layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
        return AL

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

    def initialize_multilayer_weights(self, net_dims):
        '''
        Initializes the weights of the multilayer network

        Inputs:
            net_dims - tuple of network dimensions

        Returns:
            dictionary of parameters
        '''
        np.random.seed(0)
        numLayers = len(net_dims)
        parameters = {}

        for l in range(numLayers - 1):
            # parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * 0.01
            # parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1) * 0.01
        # TODO: get weights from denoiser
            parameters["W" + str(l + 1)] = np.random.randn(net_dims[l + 1], net_dims[l]) * np.sqrt(2.0 / net_dims[l])
            parameters["b" + str(l + 1)] = np.random.randn(net_dims[l + 1], 1) * np.sqrt(2.0 / net_dims[l])

            # parameters["W" + str(l + 1)] = np.random.normal(0, np.sqrt(2.0 / net_dims[l]), (net_dims[l + 1], net_dims[l]))
            # parameters["b" + str(l + 1)] = np.random.normal(0, np.sqrt(2.0 / net_dims[l]), (net_dims[l + 1], 1))


        return parameters

    def train(self, train_data, train_label):

        num_iterations = 1000
        num_dimensions = [400,300,100,10]
        parameters = self.initialize_multilayer_weights(num_dimensions)
        A= train_data
        learning_rate = 0.2
        for ii in range(num_iterations):
            AL, caches = self.multi_layer_forward(A, parameters)
            A = np.exp(AL - np.max(AL, axis=0)) / np.sum(np.exp(AL - np.max(AL, axis=0)), axis=0, keepdims=True)
            loss = 0
            for i in range(0, train_label.shape[1]):
                index = int(train_label[0][i])
                loss = loss + np.log(A[index][i])
            loss = -loss / train_label.shape[1]
            one_hot_vector = np.zeros((train_label.size, int(train_label.max()) + 1))
            one_hot_vector[np.arange(train_label.size), train_label.astype(int)] = 1
            one_hot_vector = one_hot_vector.T
            dZ = A - one_hot_vector
            dZ = dZ / A.shape[1]
            dW = np.dot(dZ, A.T)
            db = np.sum(dZ, axis=1, keepdims=True)

            parameters["W" + str(len(num_dimensions)-1)] = parameters["W" + str(len(num_dimensions)-1)] - (
                        learning_rate * dW)
            parameters["b" + (len(num_dimensions)-1)] = parameters["b" + (len(num_dimensions)-1)] - (
                        learning_rate * db)

            print(loss)

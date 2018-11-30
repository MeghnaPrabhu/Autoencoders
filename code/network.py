'''
Python 3.x
This file implements a multi layer neural network for a multiclass classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
from activations import Activations


def initialize_weights(net_dims, network_type):
    '''
    Initializes the weights of the multilayer network

    Inputs:
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    numLayers = len(net_dims)
    parameters = {}
    if network_type == 'DAE':
        parameters["W1"] = np.random.normal(0, np.sqrt(2.0 / net_dims[0]), (net_dims[1], net_dims[0]))
        parameters["b1"] = np.random.normal(0, np.sqrt(2.0 / net_dims[0]), (net_dims[1], 1))
        parameters["W2"] = parameters["W1"].T
        # TODO: check if b1 can be initialized randomly.
        parameters["b2"] = np.random.normal(0, np.sqrt(2.0 / net_dims[0]), (net_dims[2], 1))
    else:
        for l in range(numLayers - 1):
            parameters["W" + str(l + 1)] = np.random.normal(0, np.sqrt(2.0 / net_dims[l]),
                                                            (net_dims[l + 1], net_dims[l]))
            parameters["b" + str(l + 1)] = np.random.normal(0, np.sqrt(2.0 / net_dims[l]), (net_dims[l + 1], 1))
    return parameters


def linear_forward(A, W, b):
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
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache


def layer_forward(A_prev, W, b, activation):
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
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = Activations.relu(Z)
    elif activation == "linear":
        A, act_cache = Activations.linear(Z)
    elif activation == 'sigmoid':
        A, act_cache = Activations.sigmoid(Z)
    elif activation == 'tanh':
        A, act_cache = Activations.tanh(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache


def multi_layer_forward(X, parameters):
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
    caches = []
    for l in range(1, L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "tanh")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    return AL, caches


def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    dW = (np.dot(dZ, cache['A'].T))
    db = (np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = Activations.sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = Activations.tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = Activations.relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = Activations.linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs:
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "tanh"
    for l in reversed(range(1, L + 1)):
        dA, gradients["dW" + str(l)], gradients["db" + str(l)] = \
            layer_backward(dA, caches[l - 1], parameters["W" + str(l)], parameters["b" + str(l)], activation)
        activation = "sigmoid"

    return gradients


def classify(X, Y, parameters):
    '''
    Network prediction for inputs X

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    AL, caches = multi_layer_forward(X, parameters)
    Ypred, cache, cost = Activations.softmax_cross_entropy_loss(AL, Y)
    Ypred = np.argmax(Ypred, axis=0)
    return Ypred


def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.01):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    # alpha = learning_rate * (1 / (1 + decay_rate * epoch))
    L = len(parameters) // 2
    for i in range(1, L + 1):
        parameters['W' + str(i)] = parameters['W' + str(i)] - (learning_rate * gradients['dW' + str(i)])
        parameters['b' + str(i)] = parameters['b' + str(i)] - (learning_rate * gradients['db' + str(i)])
    return parameters, learning_rate


def multi_layer_network(X, Y, validation_data, validation_label, net_dims, network_type, corrupted_input=None,
                        num_iterations=500, learning_rate=0.2, decay_rate=0.01):
    '''

    :param X: numpy.ndarray (n,m) of training data
    :param Y: numpy.ndarray (1,m) of training data labels
    :param validation_data: numpy.ndarray (n,m) of validation data
    :param validation_label: numpy.ndarray (1,m) of validation data labels
    :param net_dims: tuple of layer dimensions
    :param network_type: type of autoencoder
    :param corrupted_input: denoised input samples
    :param num_iterations: num of epochs to train
    :param learning_rate: step size for gradient descent
    :return:
     costs - list of costs over training
     parameters - dictionary of trained network parameters
    '''

    parameters = initialize_weights(net_dims, network_type)
    costs = []
    validation_costs = []
    for ii in range(num_iterations):
        # Forward Prop
        AL, caches = multi_layer_forward(corrupted_input, parameters)
        if network_type == 'DAE':
            # denoising autoencoder so use MSE
            cost = Activations.mean_squared_error(AL, X)
            dz = Activations.mean_squared_error_der(X, AL)
        else:
            # stacked autoencoder
            VL, validation_caches = multi_layer_forward(validation_data, parameters)
            A, cache, cost = Activations.softmax_cross_entropy_loss(AL, Y)
            validation_prediction, validation_cache, validation_cost = \
                Activations.softmax_cross_entropy_loss(VL, validation_label)
            dz = Activations.softmax_cross_entropy_loss_der(Y, cache)

        # Backward Prop
        gradients = multi_layer_backward(dz, caches, parameters)
        parameters, alpha = update_parameters(parameters, gradients, ii, learning_rate, decay_rate)
        if ii % 10 == 0:
            costs.append(cost)
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))
            if network_type != 'DAE':
                validation_costs.append(validation_cost)
                print("Validation Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, validation_cost, alpha))

        if ii % 100 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))
            if network_type != 'DAE':
                validation_costs.append(validation_cost)
                print("Validation Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, validation_cost, alpha))
    return costs, validation_costs, parameters

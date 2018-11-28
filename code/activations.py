import numpy as np


class Activations:

    @staticmethod
    def linear(Z):
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
        cache = {}
        return A, cache

    @staticmethod
    def linear_der(dA, cache):
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

    @staticmethod
    def sigmoid(Z):
        '''
        computes sigmoid activation of Z

        Inputs:
            Z is a numpy.ndarray (n, m)

        Returns:
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''
        A = 1 / (1 + np.exp(-Z))
        cache = {}
        cache["Z"] = Z
        return A, cache

    @staticmethod
    def sigmoid_der(dA, cache):
        '''
        computes derivative of sigmoid activation

        Inputs:
            dA is the derivative from subsequent layer. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}, where Z was the input
            to the activation layer during forward propagation

        Returns:
            dZ is the derivative. numpy.ndarray (n,m)
        '''
        dZ = np.multiply(dA, np.multiply(Activations.sigmoid(cache["Z"])[0], 1 - Activations.sigmoid(cache["Z"])[0]))
        return dZ

    @staticmethod
    def relu(Z):
        '''
        computes relu activation of Z

        Inputs:
            Z is a numpy.ndarray (n, m)

        Returns:
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''
        A = np.maximum(0, Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    @staticmethod
    def relu_der(dA, cache):
        '''
        computes derivative of relu activation

        Inputs:
            dA is the derivative from subsequent layer. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}, where Z was the input
            to the activation layer during forward propagation

        Returns:
            dZ is the derivative. numpy.ndarray (n,m)
        '''
        dZ = np.array(dA, copy=True)
        Z = cache["Z"]
        dZ[Z < 0] = 0
        return dZ

    @staticmethod
    def tanh(Z):
        '''
        computes tanh activation of Z

        Inputs:
            Z is a numpy.ndarray (n, m)

        Returns:
            A is activation. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}
        '''
        A = np.tanh(Z)
        cache = {}
        cache["Z"] = Z
        return A, cache

    @staticmethod
    def tanh_der(dA, cache):
        '''
        computes derivative of tanh activation

        Inputs:
            dA is the derivative from subsequent layer. numpy.ndarray (n, m)
            cache is a dictionary with {"Z", Z}, where Z was the input
            to the activation layer during forward propagation

        Returns:
            dZ is the derivative. numpy.ndarray (n,m)
        '''
        Z = cache["Z"]
        dZ = dA * (1 - (np.tanh(Z) ** 2))
        return dZ

    @staticmethod
    def softmax_cross_entropy_loss(Z, Y=np.array([])):
        '''
        Computes the softmax activation of the inputs Z
        Estimates the cross entropy loss

        Inputs:
            Z - numpy.ndarray (n, m)
            Y - numpy.ndarray (1, m) of labels
                when y=[] loss is set to []

        Returns:
            A - numpy.ndarray (n, m) of softmax activations
            cache -  a dictionary to store the activations later used to estimate derivatives
            loss - cost of prediction
        '''
        cache = {}
        A = np.exp(Z - np.max(Z, axis=0)) / np.sum(np.exp(Z - np.max(Z, axis=0)), axis=0, keepdims=True)
        cache['A'] = A
        one_hot_targets = np.array([np.eye(Z.shape[0])[int(Y[0][int(i)])] for i in range(len(Y[0]))]).T
        loss = -np.sum(one_hot_targets * np.log(A)) / Y.shape[1]
        return A, cache, loss

    @staticmethod
    def softmax_cross_entropy_loss_der(Y, cache):
        '''
        Computes the derivative of softmax activation and cross entropy loss

        Inputs:
            Y - numpy.ndarray (1, m) of labels
            cache -  a dictionary with cached activations A of size (n,m)

        Returns:
            dZ - numpy.ndarray (n, m) derivative for the previous layer
        '''
        one_hot_targets = np.array([np.eye(cache['A'].shape[0])[int(Y[0][int(i)])] for i in range(Y.shape[1])]).T
        dZ = cache['A'] - one_hot_targets
        return dZ / cache['A'].shape[1]

    @staticmethod
    def mean_squared_error(Z, Y):
        mse = np.mean((Z - Y) ** 2)
        return mse

import numpy as np
from numpy.random import uniform


class Layer:
    def __init__(self, w_shape, b_shape, activation, activation_name):
        self.W = w_shape
        self.b = b_shape
        self.activation = activation
        self.activation_name = activation_name


class NeuralNet:
    ACTIVATIONS = {
        'relu': lambda x: np.maximum(x, 0),
        'tanh': np.tanh,
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'linear': lambda x: x,
    }

    def __init__(self, input_len):
        self.layers = []
        self.output_len = input_len

    def add_layer(self, output_len, activation):
        layer = Layer((self.output_len, output_len), (1, output_len), NeuralNet.ACTIVATIONS[activation], activation)
        self.layers.append(layer)
        self.output_len = output_len
        return self

    def predict(self, inputs, weights_by_layer, biases_by_layer):
        inputs = np.asarray([inputs])
        for layer, weights, biases in zip(self.layers, weights_by_layer, biases_by_layer):
            inputs = layer.activation(inputs.dot(weights) + biases)
        return inputs[0]

    def random_matrix(self, func, init_limits):
        weights = []
        for layer in self.layers:
            shape = func(layer)
            inits = uniform(init_limits[0], init_limits[1], shape)
            weights.append(inits)

        return weights

    def random_weights_list(self, init_limits):
        return self.random_matrix(lambda layer: layer.W, init_limits)

    def random_biases_list(self, init_limits):
        return self.random_matrix(lambda layer: layer.b, init_limits)

    def save(self, path, weights, biases):
        with open(path, 'w') as f:
            f.write(str(len(self.layers)) + '\n')
            for layer, w, b in zip(self.layers, weights, biases):
                b = b[0]
                f.write(str(layer.activation_name) + '\n')
                f.write(' '.join([str(dim) for dim in w.shape]) + '\n')
                for w_row in w:
                    f.write(' '.join([str(val) for val in w_row]) + '\n')
                f.write(' '.join([str(dim) for dim in b.shape]) + '\n')
                f.write(' '.join([str(val) for val in b]) + '\n')

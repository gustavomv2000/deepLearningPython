import numpy as np
import math

class Neural(object):
    def __init__(self):
        self.feedForward([1, 0.9, 0.1, 0.2], [3, 2, 2], [1, 0])

    def feedForward(self, data_input=[], layers=[], desired_output=[]):
        print("How many layers: ", len(layers))

        self.current_activation = data_input
        self.activation_layer_len = len(data_input)
        self.out_layer_len = layers[0]
        self.vector_matrix = []
        self.vector_bias = []
        self.vector_output = []

        for layer in range(0, len(layers)):
            weights_matrix = np.random.randn(self.out_layer_len, self.activation_layer_len)
            self.vector_matrix.append(weights_matrix)

            bias_array = np.random.randn(self.out_layer_len)
            self.vector_bias.append(bias_array)

            next_activation_layer = bias_array

            for i in range(0, self.activation_layer_len):
                for j in range(0, self.out_layer_len):
                    next_activation_layer[j] = next_activation_layer[j] + (weights_matrix[j][i] * self.current_activation[i])

            next_activation_layer = self.sigmoidArray(next_activation_layer)
            self.vector_output.append(next_activation_layer)
            #print("next: ", next_activation_layer)

            if layer != len(layers)-1:
                self.current_activation = next_activation_layer
                self.activation_layer_len = len(next_activation_layer)
                self.out_layer_len = layers[layer+1]
        print(next_activation_layer)
        self.checkError(data_input, desired_output, next_activation_layer, layers)

    def sigmoidArray(self, data, deriv=False):
        if (deriv == True):
            return data * (1 - data)
        return 1/(1 + np.exp(-data))

    def checkError(self, data_input, desired_output, output, layers):
        self.output_error = desired_output - output
        print("output error: ", self.output_error)
        self.output_delta = self.output_error * self.sigmoidArray(output, deriv=True)
        print("output delta: ", self.output_delta)

        hidden_layers_error = []

        i_delta = self.output_delta
        self.hidden_errors = []
        self.hidden_deltas = []
        #print("MATRIX: ", self.vector_matrix)
        for i in range(len(layers)-1, 0, -1):
            print("T vector matrix i: ", self.vector_matrix[i].T)
            print("delta: ", i_delta)
            self.hidden_errors.append(i_delta.dot(self.vector_matrix[i].reshape(len(i_delta),-1)))
            print("error iteration: ", self.hidden_errors[(len(layers)-1)-i])
            print("sigm: ", self.sigmoidArray(self.vector_output[i], deriv=True))
            self.hidden_deltas.append(self.hidden_errors[(len(layers)-1)-i].reshape(1,-1) * self.sigmoidArray(self.vector_output[i], deriv=True).reshape(-1,1))
            print("delta iteration: ", self.hidden_deltas[(len(layers)-1)-i])
            i_delta = self.hidden_deltas[(len(layers)-1)-i]

        print("Hidden errors: ", self.hidden_errors)
        print("Hidden deltas: ", self.hidden_deltas)

NN = Neural()

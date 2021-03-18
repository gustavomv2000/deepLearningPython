import numpy as np
import random
import datetime

class Neural(object):
    def __init__(self):
        self.vector_matrix = []
        self.vector_bias = []
        self.vector_output = []
        self.activation_layer_len = input_size
        self.out_layer_len = layers[0]

        for layer in range(0, len(layers)):
            weights_matrix = np.random.randn(self.activation_layer_len, self.out_layer_len)
            #weights_matrix = np.ones((self.activation_layer_len, self.out_layer_len))
            self.vector_matrix.append(weights_matrix)

            bias_array = np.random.randn(self.out_layer_len)
            #bias_array = np.ones(self.out_layer_len)
            self.vector_bias.append(bias_array)

            if layer != len(layers) - 1:
                self.out_layer_len = layers[layer+1]
                self.activation_layer_len = layers[layer]

        print("Initial Weights: ", self.vector_matrix)
        print("Initial Bias: ", self.vector_bias)

    def feed_forward(self, data_input=[], layers_size=[], desired_output=[]):
        self.current_activation = data_input.copy()
        self.vector_output = []
        self.vector_output.append(data_input)

        for layer in range(0, len(layers_size)):
            weights_matrix = self.vector_matrix[layer].copy()

            next_activation_layer = np.array(self.vector_bias[layer])

            for j in range(0, len(next_activation_layer)):
                for i in range(0, len(self.current_activation)):
                    next_activation_layer[j] = np.add(next_activation_layer[j], np.dot(self.current_activation[i],
                                                                                       weights_matrix[i, j]))

            next_activation_layer = self.sigmoid_array(next_activation_layer)

            self.vector_output.append(next_activation_layer)

            self.current_activation = next_activation_layer.copy()

    def sigmoid_array(self, data, deriv=False):
        if deriv == True:
            return data * (1 - data)
        return 1/(1 + np.exp(-data))

    def calculate_errors(self, output, expected):
        return np.subtract(expected, output)

    def calculate_derivatives(self, output):
        return self.sigmoid_array(output, deriv=True)

    def calculate_deltas(self, errors, derivatives):
        return lr * (errors * (derivatives))

    def adjust_weights(self, weights, activation, deltas):
        adj = []

        if len(deltas) != 1:
            for d in range(0, len(deltas)):
                adj.append(deltas[d] * activation)
        else:
            adj.append(deltas * activation)

        for i in range(0, len(weights)):
            for j in range(0, len(weights[i])):
                weights[i][j] = weights[i][j] + adj[j][i]

        return weights

    def adjust_bias(self, bias, delta):
        return np.add(bias, lr * delta)

    def backpropation(self, vector_output, expected, data):
        output_error = self.calculate_errors(vector_output[len(vector_output)-1], expected)
        output_derivatives = self.calculate_derivatives(vector_output[len(vector_output)-1])
        output_delta = self.calculate_deltas(output_error, output_derivatives)

        self.vector_matrix[len(self.vector_matrix) - 1] = self.adjust_weights(self.vector_matrix[len(self.vector_matrix) - 1].copy(),
                                                                              vector_output[len(vector_output) - 2],
                                                                              output_delta)

        self.vector_bias[len(self.vector_bias) - 1] = self.adjust_bias(self.vector_bias[len(self.vector_bias) - 1],
                                                                       output_delta)
        prev_delta = output_delta
        for k in range(len(layers) - 1, 0, -1):
            hidden_errors = prev_delta.dot(self.vector_matrix[k].T)
            hidden_derivatives = self.calculate_derivatives(self.vector_output[k])
            hidden_deltas = self.calculate_deltas(hidden_errors, hidden_derivatives)
            self.vector_matrix[k - 1] = self.adjust_weights(self.vector_matrix[k-1].copy(),
                                                            np.array(self.vector_output[k - 1]),
                                                            hidden_deltas)

            self.vector_bias[k - 1] = self.adjust_bias(self.vector_bias[k - 1], hidden_deltas)
            prev_delta = hidden_deltas


    def train_neural(self, data_inputs):
        i = 0
        for data in data_inputs:
            for k in range(0, 1):
                self.feed_forward(data[0], layers, data[1])

                self.backpropation(self.vector_output, data[1], data[0]);

            i += 1

'''
input_size = 2
layers = [2, 1]
lr = 1
NN = Neural()
data_in = [[[1, 0], [1]]]
NN.train_neural(data_in)

print("after weights: ", NN.vector_matrix)
print("after bias: ", NN.vector_bias)
print("output: ", NN.vector_output[len(NN.vector_output) - 1])
'''

'''
input_size = 2
layers = [3, 2, 3, 1]
lr = 0.3
NN = Neural()

insert = [
    [np.array([0, 0]), [0]],
    [np.array([1, 1]), [0]],
    [np.array([1, 0]), [1]],
    [np.array([0, 1]), [1]]
]

for i in range(0, 50000):
    random.shuffle(insert)
    NN.train_neural(insert)


print("after weights: ", NN.vector_matrix)
print("after bias: ", NN.vector_bias)

NN.feed_forward([1, 0], layers, [10000])
print(NN.vector_output[len(NN.vector_output)-1])
NN.feed_forward([0, 1], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
NN.feed_forward([0, 0], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
NN.feed_forward([1, 1], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
'''

#'''
input_size = 2
layers = [2, 1]
lr = 0.4
NN = Neural()

insert = [
    [np.array([0, 1]), [0]],
    [np.array([0.1, 0.8]), [0]],
    [np.array([1, 0]), [1]],
    [np.array([1, 0.8]), [1]],
    [np.array([0.1, 0.2]), [0]],
    [np.array([0.2, 0.1]), [1]],
    [np.array([0.03, 0.02]), [1]],
    [np.array([0.02, 0.03]), [0]],
    [np.array([1, 0.9]), [1]],
    [np.array([0.9, 0.1]), [1]]
]

for i in range(0, 2000):
    random.shuffle(insert)
    NN.train_neural(insert)


print("after weights: ", NN.vector_matrix)
print("after bias: ", NN.vector_bias)

NN.feed_forward([1, 0], layers, [10000])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.7, 0.1], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([1, 0.9], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.1, 0.05], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.02, 0.01], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.2, 0.9], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.4, 0.6], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.3, 0.6], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.1, 0.3], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
NN.feed_forward([0.01, 0.02], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0]))
#'''

'''
input_size = 2
layers = [2, 3, 1]
lr = 0.3
NN = Neural()

insert = [
    [np.array([0, 1]), [1]],
    [np.array([0, 0.1]), [0.1]],
    [np.array([0, 0.2]), [0.2]],
    [np.array([0, 0.3]), [0.3]],
    [np.array([0, 0.4]), [0.4]],
    [np.array([0, 0.5]), [0.5]],
    [np.array([0, 0.6]), [0.6]],
    [np.array([0, 0.7]), [0.7]],
    [np.array([0, 0.8]), [0.8]],
    [np.array([0, 0.9]), [0.9]],
    [np.array([0.1, 0]), [0.1]],
    [np.array([0.2, 0]), [0.2]],
    [np.array([0.3, 0]), [0.3]],
    [np.array([0.4, 0]), [0.4]],
    [np.array([0.5, 0]), [0.5]],
    [np.array([0.6, 0]), [0.6]],
    [np.array([0.7, 0]), [0.7]],
    [np.array([0.8, 0]), [0.8]],
    [np.array([0.9, 0]), [0.9]],
    [np.array([0.1, 0.1]), [0.2]],
    [np.array([0.1, 0.2]), [0.3]],
    [np.array([0.1, 0.3]), [0.4]],
    [np.array([0.1, 0.4]), [0.5]],
    [np.array([0.1, 0.5]), [0.6]],
    [np.array([0.1, 0.6]), [0.7]],
    [np.array([0.1, 0.7]), [0.8]],
    [np.array([0.1, 0.8]), [0.9]],
    [np.array([0.1, 0.9]), [1]],
    [np.array([0.05, 0.1]), [0.15]],
    [np.array([0.02, 0.03]), [0.05]],
    [np.array([1, 0]), [1]],
]

for i in range(0, 10000):
    random.shuffle(insert)
    NN.train_neural(insert)


print("after weights: ", NN.vector_matrix)
print("after bias: ", NN.vector_bias)

NN.feed_forward([1, 0], layers, [10000])
print(round(NN.vector_output[len(NN.vector_output)-1][0] * 10, 2))
NN.feed_forward([0.7, 0.1], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0] * 10, 2))
NN.feed_forward([0, 0.9], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0] * 10, 2))
NN.feed_forward([0.1, 0.2], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0] * 10, 2))
NN.feed_forward([0.05, 0.05], layers, [11111])
print(round(NN.vector_output[len(NN.vector_output)-1][0] * 10, 2))
'''

'''
input_size = 2
layers = [4, 11]
lr = 0.3
NN = Neural()

insert = [
    [np.array([0, 1]), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0, 0.1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
    [np.array([0, 0.2]), [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    [np.array([0, 0.3]), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    [np.array([0, 0.4]), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    [np.array([0, 0.5]), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    [np.array([0, 0.6]), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    [np.array([0, 0.7]), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0, 0.8]), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0, 0.9]), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.1, 0]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
    [np.array([0.2, 0]), [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    [np.array([0.3, 0]), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    [np.array([0.4, 0]), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    [np.array([0.5, 0]), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    [np.array([0.6, 0]), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    [np.array([0.7, 0]), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.8, 0]), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.9, 0]), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.1, 0.1]), [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    [np.array([0.1, 0.2]), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    [np.array([0.1, 0.3]), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    [np.array([0.1, 0.4]), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    [np.array([0.1, 0.5]), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    [np.array([0.1, 0.6]), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.1, 0.7]), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.1, 0.8]), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.1, 0.9]), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.2, 0.1]), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    [np.array([0.3, 0.1]), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    [np.array([0.4, 0.1]), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    [np.array([0.5, 0.1]), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    [np.array([0.6, 0.1]), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.7, 0.1]), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.8, 0.1]), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.9, 0.1]), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.2, 0.2]), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    [np.array([0.2, 0.3]), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    [np.array([0.2, 0.4]), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    [np.array([0.2, 0.5]), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.2, 0.6]), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.2, 0.7]), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.8, 0.2]), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.2, 0.2]), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    [np.array([0.3, 0.2]), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    [np.array([0.4, 0.2]), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    [np.array([0.5, 0.2]), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.6, 0.2]), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.7, 0.2]), [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0.8, 0.2]), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [np.array([0, 0]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
]

time_start = datetime.datetime.now()

for i in range(0, 3000):
    random.shuffle(insert)
    NN.train_neural(insert)


print("after weights: ", NN.vector_matrix)
print("after bias: ", NN.vector_bias)

NN.feed_forward([1, 0], layers, [10000])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0, 0.9], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.7, 0.1], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.3, 0.4], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.3, 0.3], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.2, 0.3], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.1, 0.3], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.1, 0.2], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0, 0.2], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.1, 0], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))
NN.feed_forward([0.05, 0.05], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
print(10 - np.argmax(NN.vector_output[len(NN.vector_output)-1]))

time_finish = datetime.datetime.now();
print('--------------------------------------------')
print("TOTAL RUNTIME: ", time_finish - time_start)
'''

'''
input_size = 2
layers = [2, 2]
lr = 0.3
NN = Neural()

insert = [
    [np.array([0, 1]), [1, 0]],
    [np.array([1, 0]), [1, 0]],
    [np.array([0, 0]), [0, 1]],
    [np.array([1, 1]), [0, 1]],
]

time_start = datetime.datetime.now()

for i in range(0, 10000):
    random.shuffle(insert)
    NN.train_neural(insert)


print("after weights: ", NN.vector_matrix)
print("after bias: ", NN.vector_bias)

NN.feed_forward([1, 0], layers, [10000])
print(NN.vector_output[len(NN.vector_output)-1])
NN.feed_forward([0, 1], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
NN.feed_forward([0, 0], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])
NN.feed_forward([1, 1], layers, [11111])
print(NN.vector_output[len(NN.vector_output)-1])

time_finish = datetime.datetime.now();
print('--------------------------------------------')
print("TOTAL RUNTIME: ", time_finish - time_start)
'''

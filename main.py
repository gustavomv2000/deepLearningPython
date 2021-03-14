import numpy as np

def createNeuralNetwork(data_input=[], layers=[]):
    print("How many layers: ", len(layers))

    activation_layer_len = len(data_input)
    out_layer_len = layers[0]

    weights_matrix = np.random.randn(out_layer_len, activation_layer_len)
    print("weights: ", weights_matrix)

    bias_array = np.random.randn(out_layer_len)
    print("bias: ", bias_array)

    next_activation_layer = bias_array

    for i in range(0, activation_layer_len):
        for j in range(0, out_layer_len):
            next_activation_layer[j] = next_activation_layer[j] + (weights_matrix[j][i] * data_input[i])

    print("next: ", next_activation_layer)


createNeuralNetwork([1,2,3,4], [3,2,2])
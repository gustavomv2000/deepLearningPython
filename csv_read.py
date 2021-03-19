import csv
import main as IA
import random
import datetime
import numpy as np

data_inputs = []

with open('stocks - stocks (1).csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    csv_output = []
    input_float = []

    for row in csv_reader:
        csv_output.append(row)
        input_float.append(float(row[1]))

    max_value = max(input_float)

    for row in csv_output:
        input_expected = []

        #print(f'Data: \t{row[0]} valor: {row[1]}')

        if line_count > 30:
            #print("line count: ", line_count)
            prev_data = []
            for i in range(30, 0, -1):
                #print('Last data: ', csv_output[line_count - i][1])
                prev_data.append(float(csv_output[line_count - i][1])/max_value)
            input_expected.append(prev_data)
            input_expected.append(float(row[1])/max_value)
            data_inputs.append(input_expected)

        line_count += 1

    #print(f'Processed {line_count} lines.')
    #print('normalized types: ', data_inputs)
    #print('normalized types: ', max(input_float))


time_start = datetime.datetime.now()

input_size = 30
layers = [8, 2, 1]
lr = 0.05
NN = IA.Neural(input_size, layers, lr)

input_test = [11.67, 11.59, 11.59, 11.59, 11.56, 11.38, 11.39, 11.73, 11.73, 11.73, 11.44, 11.46, 11.71, 12.16, 12.13,
              12.13, 12.13, 11.85, 11.94, 11.81, 11.92, 11.61, 11.61, 11.61, 11.63, 11.45, 11.27, 11.04, 10.88, 10.61] #10.79
input_test = [x / max_value for x in input_test]

input_test_2 = [11.59, 11.59, 11.59, 11.56, 11.38, 11.39, 11.73, 11.73, 11.73, 11.44, 11.46, 11.71, 12.16, 12.13, 12.13,
                12.13, 11.85, 11.94, 11.81, 11.92, 11.61, 11.61, 11.61, 11.63, 11.45, 11.27, 11.04, 10.88, 10.61, 10.79] #10.81
input_test_2 = [x / max_value for x in input_test_2]

input_test_3 = [11.59, 11.59, 11.56, 11.38, 11.39, 11.73, 11.73, 11.73, 11.44, 11.46, 11.71, 12.16, 12.13, 12.13, 12.13,
                11.85, 11.94, 11.81, 11.92, 11.61, 11.61, 11.61, 11.63, 11.45, 11.27, 11.04, 10.88, 10.61, 10.79, 10.81] #10.83
input_test_3 = [x / max_value for x in input_test_3]

input_test_4 = [11.59, 11.56, 11.38, 11.39, 11.73, 11.73, 11.73, 11.44, 11.46, 11.71, 12.16, 12.13, 12.13, 12.13, 11.85,
                11.94, 11.81, 11.92, 11.61, 11.61, 11.61, 11.63, 11.45, 11.27, 11.04, 10.88, 10.61, 10.79, 10.81, 10.83] #10.83
input_test_4 = [x / max_value for x in input_test_4]

input_test_5 = [11.56, 11.38, 11.39, 11.73, 11.73, 11.73, 11.44, 11.46, 11.71, 12.16, 12.13, 12.13, 12.13, 11.85, 11.94,
                11.81, 11.92, 11.61, 11.61, 11.61, 11.63, 11.45, 11.27, 11.04, 10.88, 10.61, 10.79, 10.81, 10.83, 10.83] #10.77
input_test_5 = [x / max_value for x in input_test_5]

#'''
while NN.lr <= 1.02:
    NN = IA.Neural(input_size, layers, NN.lr)
    error_final = 0
    for i in range(0, 100):
        random.shuffle(data_inputs)
        NN.train_neural(data_inputs)

    NN.feed_forward(input_test, layers)
    error_final += abs(NN.vector_output[len(NN.vector_output)-1][0]*max_value - 10.79)
    NN.feed_forward(input_test_2, layers)
    error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.81)
    NN.feed_forward(input_test_3, layers)
    error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.83)
    NN.feed_forward(input_test_4, layers)
    error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.83)
    NN.feed_forward(input_test_5, layers)
    error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.77)

    print('--------------------------------------------------------------')
    #print('Returned value: ', NN.vector_output[len(NN.vector_output)-1][0]*max_value, ' expected: 10.79')
    #print('Error: ', error_final)
    print('ERROR: ', error_final/5)
    print('learning rate: ', NN.lr)
    NN.lr += 0.05
#'''

'''
error_final = 0
for i in range(0, 20):
    random.shuffle(data_inputs)
    NN.train_neural(data_inputs)


NN.feed_forward(input_test, layers)
error_final += abs(NN.vector_output[len(NN.vector_output)-1][0]*max_value - 10.79)
print('---------------------------------------------------------------------------------------')
print('Expected value: 10,79 // Actual returned value: ', NN.vector_output[len(NN.vector_output)-1]*max_value)
print('The error was: ', error_final)
error_final = 0
NN.feed_forward(input_test_2, layers)
error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.81)
print('---------------------------------------------------------------------------------------')
print('Expected value: 10,81 // Actual returned value: ', NN.vector_output[len(NN.vector_output)-1]*max_value)
print('The error was: ', error_final)
error_final = 0
NN.feed_forward(input_test_3, layers)
error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.83)
print('---------------------------------------------------------------------------------------')
print('Expected value: 10,83 // Actual returned value: ', NN.vector_output[len(NN.vector_output)-1]*max_value)
print('The error was: ', error_final)
error_final = 0
NN.feed_forward(input_test_4, layers)
error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.83)
print('---------------------------------------------------------------------------------------')
print('Expected value: 10,83 // Actual returned value: ', NN.vector_output[len(NN.vector_output)-1]*max_value)
print('The error was: ', error_final)
error_final = 0
NN.feed_forward(input_test_5, layers)
error_final += abs(NN.vector_output[len(NN.vector_output) - 1][0] * max_value - 10.77)
print('---------------------------------------------------------------------------------------')
print('Expected value: 10,77 // Actual returned value: ', NN.vector_output[len(NN.vector_output)-1]*max_value)
print('The error was: ', error_final)
error_final = 0
'''

time_finish = datetime.datetime.now();
print('--------------------------------------------')
print("TOTAL RUNTIME: ", time_finish - time_start)

print('--------------------------------------------------------------------')
#print("after weights: ", NN.vector_matrix)
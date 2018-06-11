import numpy as np

nx = 5
layer_dims = [nx, 4, 3, 2, 1]

parameters = {}

for i in range(1, len(layer_dims)):
    parameters['w' + str(i)] = np.random.rand(layer_dims[i], layer_dims[i - 1]) * 0.01
    parameters['b' + str(i)] = np.random.rand(layer_dims[i], 1) * 0.01

for i in range(1, len(layer_dims)):
    print('Layer ' + str(i) + ' parameters:')
    print(parameters['w' + str(i)].shape)
    print(parameters['b' + str(i)].shape)
    # print('w' + str(i) + ':', parameters['w' + str(i)])
    # print('b' + str(i) + ':', parameters['b' + str(i)])

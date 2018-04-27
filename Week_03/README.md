# One hidden layer Neural Network

## 1 Neural Network Overview

![neural_network_overview](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/neural_network_overview.png)

## 2 Neural Network Representation

![neural_network_representation](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/neural_network_representation.png)

## 3 Computing a Neural Network's Output

![computing_a_neural_network_output_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/computing_a_neural_network_output_01.png)

![computing_a_neural_network_output_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/computing_a_neural_network_output_02.png)

## 4 Vectorizing across multiple examples

![vectorizing_across_multiple_examples](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/vectorizing_across_multiple_examples.png)

## 5 Explanation for vectorized implementation

![explanation_for_vectorized_implementation](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/explanation_for_vectorized_implementation.png)

## 6 Activation functions

![activation_functions](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/activation_functions.png)

###sigmoid

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

x = np.linspace(-10,10)
y = sigmoid(x)

plt.plot(x,y,label='sigmoid',color='blue')
plt.show()
```

## Practice Questions

<h3> 1 What is the "cache" used for in our implementation of forward propagation and backward propagation?</h3>

- A We use it to pass variables computed during forward propagation to the corresponding backward propagation step. It contains useful values for backward propagation to compute derivatives.
- B We use it to pass variables computed during backward propagation to the corresponding forward propagation step. It contains useful values for forward propagation to compute activations.
- C It is used to cache the intermediate values of the cost function during training.
- D It is used to keep track of the hyperparameters that we are searching over, to speed up computation.

<h3> 2 Among the following, which ones are "hyperparameters"? (Check all that apply.)</h3>

- A size of the hidden layers n<sup>[l]</sup>
- B number of layers L in the neural network
- C learning rate α
- D activation values a<sup>[l]</sup>
- E number of iterations
- F weight matrices W<sup>[l]</sup>
- G bias vectors b<sup>[l]</sup>

<h3> 3 Which of the following statements is true? </h3>

- A The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.
- B The earlier layers of a neural network are typically computing more complex features of the input than the deeper layers.

<h3> 4 Vectorization allows you to compute forward propagation in an L-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False? </h3>

- A True
- B False

<h3> 5 Assume we store the values for n<sup>[l]</sup> in an array called layers, as follows: layer_dims = [n<sub>x</sub>, 4,3,2,1]. So layer 1 has four hidden units, layer 2 has 3 hidden units and so on. Which of the following for-loops will allow you to initialize the parameters for the model? </h3>

- A
```
for(i in range(1,len(layer_dims)/2)):
    parameter['W' + str(i)] = np.random.randn(layers[i],layers[i-1]) * 0.01
    parameter['b' + str(i)] = np.random.randn(layers[i],1) * 0.01
```
- B
```
for(i in range(1,len(layer_dims)/2)):
    parameter['W' + str(i)] = np.random.randn(layers[i],layers[i-1]) * 0.01
    parameter['b' + str(i)] = np.random.randn(layers[i-1],1) * 0.01
```
- C
```
for(i in range(1,len(layer_dims))):
    parameter['W' + str(i)] = np.random.randn(layers[i-1],layers[i]) * 0.01
    parameter['b' + str(i)] = np.random.randn(layers[i],1) * 0.01
```
- D
```
for(i in range(1,len(layer_dims))):
    parameter['W' + str(i)] = np.random.randn(layers[i],layers[i-1]) * 0.01
    parameter['b' + str(i)] = np.random.randn(layers[i],1) * 0.01
```

<h3> 6 Consider the following neural network. How many layers does this network have?</h3>

![key_concepts_on_deep_neural_networks_06](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/key_concepts_on_deep_neural_networks_06.png)

- A The number of layers L is 4. The number of hidden layers is 3.
- B The number of layers L is 3. The number of hidden layers is 3.
- C The number of layers L is 4. The number of hidden layers is 4.
- D The number of layers L is 5. The number of hidden layers is 4.

<h3> 7 During forward propagation, in the forward function for a layer l you need to know what is the activation function in a layer (Sigmoid, tanh, ReLU, etc.). During backpropagation, the corresponding backward function also needs to know what is the activation function for layer l, since the gradient depends on it. True/False? </h3>

- A True
- B False

<h3> 8 There are certain functions with the following properties: (i) To compute the function using a shallow network circuit, you will need a large network (where we measure size by the number of logic gates in the network), but (ii) To compute it using a deep network circuit, you need only an exponentially smaller network. True/False?</h3>

- A True
- B False

<h3> 9 Consider the following 2 hidden layer neural network: Which of the following statements are True? (Check all that apply). </h3>

![key_concepts_on_deep_neural_networks_09](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/key_concepts_on_deep_neural_networks_09.png)

- A W<sup>[1]</sup> will have shape (4, 4)
- B b<sup>[1]</sup> will have shape (4, 1)
- C W<sup>[1]</sup> will have shape (3, 4)
- D b<sup>[1]</sup> will have shape (3, 1)

- E W<sup>[2]</sup> will have shape (3, 4)
- F b<sup>[2]</sup> will have shape (1, 1)
- G W<sup>[2]</sup> will have shape (3, 1)
- H b<sup>[2]</sup> will have shape (3, 1)

- I W<sup>[3]</sup> will have shape (3, 1)
- J b<sup>[3]</sup> will have shape (1, 1)
- K W<sup>[3]</sup> will have shape (1, 3)
- L b<sup>[3]</sup> will have shape (3, 1)

<h3> 10 
Whereas the previous question used a specific network, in the general case what is the dimension of W^{[l]}, the weight matrix associated with layer l? </h3>

- A W<sup>[l]</sup> has shape (n<sup>[l]</sup>,n<sup>[l−1]</sup>)
- B W<sup>[l]</sup> has shape (n<sup>[l+1]</sup>,n<sup>[l]</sup>)
- C W<sup>[l]</sup> has shape (n<sup>[l]</sup>,n<sup>[l+1]</sup>)
- D W<sup>[l]</sup> has shape (n<sup>[l−1]</sup>,n<sup>[l]</sup>)
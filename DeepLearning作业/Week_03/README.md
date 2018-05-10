## Practice Questions

<h3> 1 Which of the following are true? (Check all that apply.) </h3>

- A a<sup>[2]</sup> denotes the activation vector of the 2<sup>nd</sup> layer.
- B a<sup>\[2](12)</sup> denotes the activation vector of the 2<sup>nd</sup> layer for the 12<sup>th</sup> training example.
- C X is a matrix in which each row is one training example.
- D a<sup>\[2](12)</sup> denotes activation vector of the 12<sup>th</sup> layer on the 2<sup>nd</sup> training example.
- E X is a matrix in which each column is one training example.
- F a<sup>[2]</sup><sub>4</sub> is the activation output by the 4<sup>th</sup> neuron of the 2<sup>nd</sup> layer
- G a<sup>[2]</sup><sub>4</sub> is the activation output of the 2<sup>nd</sup> layer for the 4th training example

<h3> 2 The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False? </h3>

- A True
- B False

<h3> 3 Which of these is a correct vectorized implementation of forward propagation for layer l, where 1≤l≤L? </h3>

- A 
>* Z<sup>[l]</sup>=W<sup>[l]</sup>A<sup>[l]</sup>+b<sup>[l]</sup>
>* A<sup>[l+1]</sup>=g<sup>[l+1]</sup>(Z<sup>[l]</sup>)
- B 
>* Z<sup>[l]</sup>=W<sup>[l]</sup>A<sup>[l−1]</sup>+b<sup>[l]</sup>
>* A<sup>[l]</sup>=g<sup>[l]</sup>(Z<sup>[l]</sup>)
- C 
>* Z<sup>[l]</sup>=W<sup>[l]</sup>A<sup>[l]</sup>+b<sup>[l]</sup>
>* A<sup>[l+1]</sup>=g<sup>[l]</sup>(Z<sup>[l]</sup>)
- D 
>* Z<sup>[l]</sup>=W<sup>[l−1]</sup>A<sup>[l]</sup>+b<sup>[l−1]</sup>
>* A[l]=g<sup>[l]</sup>(Z<sup>[l]</sup>)

<h3> 4 You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer? </h3>

- A ReLU
- B Leaky ReLU
- C sigmoid
- D tanh

<h3> 5 Consider the following code: What will be B.shape? (If you’re not sure, feel free to run this in python to find out). </h3>

```
A = np.random.rand(4,3)
B = np.sum(A,axis = 1,keepdims = True)
```
- A (4, 1)
- B (4, )
- C (1, 3)
- D (, 3)

<h3> 6 Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements is true? </h3>

- A Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons.
- B Each neuron in the first hidden layer will perform the same computation in the first iteration. But after one iteration of gradient descent they will learn to compute different things because we have “broken symmetry”.
- C Each neuron in the first hidden layer will compute the same thing, but neurons in different layers will compute different things, thus we have accomplished “symmetry breaking” as described in lecture.
- D The first hidden layer’s neurons will perform different computations from each other even in the first iteration; their parameters will thus keep evolving in their own way.

<h3> 7 Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False? </h3>

- A True
- B False 

<h3> 8 You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen? </h3>

- A This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow.
- B This will cause the inputs of the tanh to also be very large, causing the units to be “highly activated” and thus speed up learning compared to if the weights had to start from small values.
- C It doesn’t matter. So long as you initialize the weights randomly gradient descent is not affected by whether the weights are large or small.
- D This will cause the inputs of the tanh to also be very large, thus causing gradients to also become large. You therefore have to set α to be very small to prevent divergence; this will slow down learning.

<h3> 9 Consider the following 1 hidden layer neural network: </h3>

![shallow_neural_networks_09](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/shallow_neural_networks_09.png)

- A W<sup>[1]</sup> will have shape (2, 4)
- B b<sup>[1]</sup> will have shape (4, 1)
- C W<sup>[1]</sup> will have shape (4, 2)
- D b<sup>[1]</sup> will have shape (2, 1)
- E W<sup>[2]</sup> will have shape (1, 4)
- F b<sup>[2]</sup> will have shape (4, 1)
- G W<sup>[2]</sup> will have shape (4, 1)
- H b<sup>[2]</sup> will have shape (1, 1)

<h3> 10 In the same network as the previous question, what are the dimensions of Z<sup>[1]</sup> and A<sup>[1]</sup>? </h3>

- A Z<sup>[1]</sup> and A<sup>[1]</sup> are (1,4)
- B Z<sup>[1]</sup> and A<sup>[1]</sup> are (4,2)
- C Z<sup>[1]</sup> and A<sup>[1]</sup> are (4,m)
- D Z<sup>[1]</sup> and A<sup>[1]</sup> are (4,1)
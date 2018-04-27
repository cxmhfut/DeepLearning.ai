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

<h3>sigmoid</h3>

![sigmoid](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/sigmoid.png)

<h3>tanh</h3>

![tanh](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/tanh.png)

<h3>relu and leaky relu</h3>

![relu](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/relu.png)

两者的优点是：
<p>第一，在z区间变动很大的情况下，激活函数的导数或者激活函数的斜率都会远大于
0，在程序实现就是一个 if-else 语句，而 sigmoid 函数需要进行浮点四则运算，在实践中，
使用 ReLu 激活函数神经网络通常会比使用 sigmoid 或者 tanh 激活函数学习的更快。</p>
<p>第二， sigmoid 和 tanh 函数的导数在正负饱和区的梯度都会接近于 0，这会造成梯度弥
散，而 Relu 和 Leaky ReLu 函数大于 0 部分都为常熟，不会产生梯度弥散现象。 (同时应该注
意到的是， Relu 进入负半区的时候，梯度为 0，神经元此时不会训练，产生所谓的稀疏性，
而 Leaky ReLu 不会有这问题)</p>
z ReLu 的梯度一半都是 0，但是，有足够的隐藏层使得 z 值大于 0，所以对大多数的
训练数据来说学习过程仍然可以很快。
快速概括一下不同激活函数的过程和结论。

- sigmoid 激活函数：除了输出层是一个二分类问题基本不会用它。
- tanh 激活函数： tanh 是非常优秀的，几乎适合所有场合。
- ReLu 激活函数：最常用的默认函数，，如果不确定用哪个激活函数，就使用 ReLu 或者
Leaky ReLu。

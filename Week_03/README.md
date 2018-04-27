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

## 7 Why do you need non-linear activation functions

如果你是用线性激活函数或者叫恒等激励函数，那么神经网络只是把输入线性组合再输
出。

我们稍后会谈到深度网络，有很多层的神经网络，很多隐藏层。事实证明，如果你使用
线性激活函数或者没有使用一个激活函数，那么无论你的神经网络有多少层一直在做的只是
计算线性函数，所以不如直接去掉全部隐藏层。在我们的简明案例中，事实证明如果你在隐
藏层用线性激活函数，在输出层用 sigmoid 函数，那么这个模型的复杂度和没有任何隐藏层
的标准 Logistic 回归是一样的，如果你愿意的话，可以证明一下。

在这里线性隐层一点用也没有，因为这两个线性函数的组合本身就是线性函数，所以除
非你引入非线性，否则你无法计算更有趣的函数，即使你的网络层数再多也不行；只有一个
地方可以使用线性激活函数------g(z)=z就是你在做机器学习中的回归问题。 y是一个实
数，举个例子，比如你想预测房地产价格，y就不是二分类任务 0 或 1，而是一个实数，从
0 到正无穷。如果y是个实数，那么在输出层用线性激活函数也许可行，你的输出也是一个
实数，从负无穷到正无穷。

总而言之，不能在隐藏层用线性激活函数，可以用 ReLU 或者 tanh 或者 leaky ReLU 或
者其他的非线性激活函数，唯一可以用线性激活函数的通常就是输出层；除了这种情况，会
在隐层用线性函数的，除了一些特殊情况，比如与压缩有关的，那方面在这里将不深入讨论。
在这之外， 在隐层使用线性激活函数非常少见。因为房价都是非负数，所以我们也可以在输
出层使用 ReLU 函数这样你的ŷ都大于等于 0。
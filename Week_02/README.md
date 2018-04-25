# Basics of Neural Network Programming

## 1 Binary Classification

Example: Cat vs Non-Cat
The goal is to train a classifier that the input is an image represented by a feature vector, x and predicts
whether the corresponding label y is 1 or 0. In this case, whether this is a cat image (1) or a non-cat image
(0).

![binary_classify_cat_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/binary_classify_cat_01.png)

二分类问题：输入一张猫的图片（64\*64\*3），输出一个标签y（0/1）用来表示图片中是否有猫。

![binary_classify_cat_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/binary_classify_cat_02.png)

An image is store in the computer in three separate matrices corresponding to the Red, Green, and Blue
color channels of the image. The three matrices have the same size as the image, for example, the
resolution of the cat image is 64 pixels X 64 pixels, the three matrices (RGB) are 64 X 64 each.
The value in a cell represents the pixel intensity which will be used to create a feature vector of ndimension. In pattern recognition and machine learning, a feature vector represents an object, in this
case, a cat or no cat.
To create a feature vector, X the pixel intensity values will be “unroll” or “reshape” for each color. The
dimension of the input feature vector X is n<sub>x</sub> = 64\*64\*3 = 12 288.

用X来表示输入的一张图片，X的大小是64\*64\*3=12 288，表示这张图片的像素值（尺寸64\*64，RGB的3个颜色通道）

m个训练样本：{(x<sup>(1)</sup>,y<sup>(1)</sup>),(x<sup>(2)</sup>,y<sup>(2)</sup>),...,(x<sup>(m)</sup>,y<sup>(m)</sup>)}

## 2 Logistic Regression

![logistic_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_01.png)

![logistic_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_02.png)

我们构造一个线性函数：ŷ = W<sup>T</sup>X + b

其中ŷ = P(y=1|X) ŷ表示输入为X的情况下，这张图片的标签y=1的概率。

而概率值是一个0~1之间的数，我们构造的函数不能保证ŷ∈[0,1]

我们利用一个sigmoid函数将我们的输出值映射到[0,1]上

## 3 Logistic Regression Cost Function

![logistic_regression_cost_function](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_regression_cost_function.png)

## 4 Gradient Descent

![gradient_descent_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/gradient_descent_01.png)

![gradient_descent_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/gradient_descent_02.png)

## 5 Derivatives & 6 More derivative examples

![derivatives_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/derivatives_01.png)

![derivatives_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/derivatives_02.png)

![derivatives_03](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/derivatives_03.png)

## 7 Computation Graph & 8 Derivatives with a Computation Graph

![computation_graph_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/computation_graph_01.png)

![computation_graph_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/computation_graph_02.png)


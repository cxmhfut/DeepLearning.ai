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

![derivatives_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/derivatives_01.jpg)

![derivatives_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/derivatives_02.png)

![derivatives_03](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/derivatives_03.png)

## 7 Computation Graph & 8 Derivatives with a Computation Graph

![computation_graph_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/computation_graph_01.jpg)

![computation_graph_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/computation_graph_02.jpg)

## 9 Logistic Regression Gradient descent & 10 Gradient descent on m examples

![logistic_regression_derivatives](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_regression_derivatives.png)

![logistic_regression_on_m_samples](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_regression_on_m_samples.png)

## 11 Vectorization & 12 More vectorization examples

向量化消除显示循环提高运算速度

1维矩阵乘法

```python
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("Vectorized version:"+str(1000*(toc-tic))+"ms")

c = 0
tic = time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc = time.time()

print(c)
print("For loop:"+str(1000*(toc-tic))+"ms")
```
```
250232.209003
Vectorized version:8.011817932128906ms
250232.209003
For loop:717.4687385559082ms
```

多维矩阵乘法 u=A*v

```python
import numpy as np
import time

#u = A*v

row1 = 1
col1 = 1000000
row2 = 1000000
col2 = 2

A = np.random.rand(row1,col1)
v = np.random.rand(row2,col2)
u = np.zeros([row1,col2])

tic = time.time()
for i in range(row1):
    for j in range(col2):
        u[i][j] = 0
        for k in range(col1):
            u[i][j]+=A[i][k]*v[k][j]
toc = time.time()

print(u)
print("For loop:"+str(1000*(toc-tic))+"ms")

tic = time.time()
u = np.dot(A,v)
toc = time.time()

print(u)
print("Vectorized version:"+str(1000*(toc-tic))+"ms")
```

```
[[ 250247.23488535  249917.57002751]]
For loop:2892.9190635681152ms
[[ 250247.23488534  249917.57002752]]
Vectorized version:51.033735275268555ms
```

u = exp(v)
```python
import numpy as np
import time

size = 1000000

v = np.random.rand(size)
u = np.zeros(size)

tic = time.time()
for i in range(size):
    u[i] = np.exp(v[i])
toc = time.time()

print(u)
print("For loop:"+str(1000*(toc-tic))+"ms")

tic = time.time()
u = np.exp(v)
toc = time.time()

print(u)
print("Vectorized version:"+str(1000*(toc-tic))+"ms")
```

```
[ 2.11975742  1.52438811  1.27205874 ...,  1.82248043  1.87373109
  1.42462108]
For loop:2436.615228652954ms
[ 2.11975742  1.52438811  1.27205874 ...,  1.82248043  1.87373109
  1.42462108]
Vectorized version:14.008522033691406ms
```

## 13 Vectorizing Logistic Regression

![vectorizing_logistic_regression](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/vectorizing_logistic_regression.png)

## 14 Vectorizing Logistic Regression's Gradient Computation

![gradient_computation](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/gradient_computation.png.png)

## 15 Broadcasting in Python

计算矩阵A每个元素的值所占当前列的百分比

```python
import numpy as np

A = np.array([[56.0,0.0,4.4,68.0],
              [1.2,104.0,52.0,8.0],
              [1.8,135.0,99.0,0.9]])

print(A)

cal = A.sum(axis=0)#axis:0 vertical 1 horizontal

print(cal)

percentage = 100 * A / cal.reshape(1,4)

print(percentage)
```

A:矩阵A

```
[[  56.     0.     4.4   68. ]
 [   1.2  104.    52.     8. ]
 [   1.8  135.    99.     0.9]]
```

cal:矩阵A各列之和

```
[  59.   239.   155.4   76.9]
```

percentage:百分比

```
[[ 94.91525424   0.           2.83140283  88.42652796]
 [  2.03389831  43.51464435  33.46203346  10.40312094]
 [  3.05084746  56.48535565  63.70656371   1.17035111]]
```

(m,n)的矩阵与一个(1,n)的矩阵相加时,(1,n)会竖直复制成(m,n)的矩阵,然后进行相关操作
(m,n)的矩阵与一个(m,1)的矩阵相加时,(m,1)会横向复制成(m,n)的矩阵,然后进行相关操作

```python
import numpy as np

A = np.array([[1,2,3],
              [4,5,6]])

B = np.array([[100,200,300]])

C = np.array([[100],
              [200]])

print(A+B)
print(A+C)
```

```
[[101 202 303]
 [104 205 306]]
[[101 102 103]
 [204 205 206]]
```

## 16 A note on python/numpy vectors

![python_numpy_vector](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/python_numpy_vector.png.png)

## 17 Quick tour of Jupyter/ipython notebooks

Jupyter

## 18 Explanation of logistic regression cost function

![logistic_regression_cost_function_01](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_regression_cost_function_01.png.png)

![logistic_regression_cost_function_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_regression_cost_function_02.png.png)
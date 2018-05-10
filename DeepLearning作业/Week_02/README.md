## Practice Questions

<h3> 1 What does a neuron compute? </h3>

- A A neuron computes an activation function followed by a linear function (z = Wx + b)
- B A neuron computes a linear function (z = Wx + b) followed by an activation function
- C A neuron computes a function g that scales the input x linearly (Wx + b)
- D A neuron computes the mean of all features before applying the output to an activation function

```
一个神经元计算的是什么
A 一个神经元计算一个激活函数，后跟一个线性函数（z = Wx + b）
B 神经元计算一个线性函数（z = Wx + b），后跟一个激活函数
C 神经元计算函数g，线性地对输入x进行缩放（Wx + b）
D 神经元在将输出应用于激活函数之前计算所有特征的平均值
```

<h5>Answer:B</h5>

<h3> 2 Which of these is the "Logistic Loss"? </h3>

- A L<sup>(i)</sup>(ŷ<sup>(i)</sup>,y<sup>(i)</sup>)=−(y<sup>(i)</sup>log(ŷ<sup>(i)</sup>)+(1−y<sup>(i)</sup>)log(1−ŷ<sup>(i)</sup>))
- B L<sup>(i)</sup>(ŷ<sup>(i)</sup>,y<sup>(i)</sup>)=∣y<sup>(i)</sup>−ŷ<sup>(i)</sup>∣
- C L<sup>(i)</sup>(ŷ<sup>(i)</sup>,y<sup>(i)</sup>)=max(0,y<sup>(i)</sup>−ŷ<sup>(i)</sup>)
- D L<sup>(i)</sup>(ŷ<sup>(i)</sup>,y<sup>(i)</sup>)=∣y<sup>(i)</sup>−ŷ<sup>(i)</sup>∣<sup>2</sup>

<h5>Answer:A</h5>

<h3> 3 Suppose img is a (32,32,3) array, representing a 32x32 image with 3 color channels red, green and blue. How do you reshape this into a column vector? </h3>

- A x = img.reshape((32\*32\*3,1))
- B x = img.reshape((32*32,3))
- C x = img.reshape((1,32\*32*3))
- D x = img.reshape((3,32*32))

<h5>Answer:C</h5>

<h3> 4 Consider the two following random arrays "a" and "b": What will be the shape of "c"?</h3>

```
a = np.random.rand(2,3) #a.shape = (2,3)
b = np.random.rand(2,1) #b.shape = (2,1)
c = a + b
```
- A c.shape = (2, 1)
- B c.shape = (3, 2)
- C c.shape = (2, 3)
- D The computation cannot happen because the sizes don't match. It's going to be "Error"!

```python
import numpy as np

a = np.random.rand(2,3)
b = np.random.rand(2,1)
c = a + b

print(c.shape)
```

```
(2, 3)
```

<h5>Answer:C</h5>

<h3> 5 Consider the two following random arrays "a" and "b": What will be the shape of "c"?</h3>

```
a = np.random.rand(4,3) #a.shape = (4,3)
b = np.random.rand(3,2) #b.shape = (3,2)
c = a * b
```
- A c.shape = (4,2)
- B c.shape = (4, 3)
- C The computation cannot happen because the sizes don't match. It's going to be "Error"!
- D c.shape = (3, 3)

```python
import numpy as np

a = np.random.rand(4,3)
b = np.random.rand(3,2)
c = a * b
print(c.shape)
```

```
Traceback (most recent call last):
    c = a * b
ValueError: operands could not be broadcast together with shapes (4,3) (3,2) 
```

<h5>Answer:C</h5>

<h3> 6 Suppose you have n<sub>x</sub> input features per example. Recall that X=[x<sup>(1)</sup>x<sup>(2)</sup>...x<sup>(m)</sup>]. What is the dimension of X? </h3>

- A (m,1)
- B (n<sub>x</sub>,m)
- C (m,n<sub>x</sub>)
- D (1,m)

<h5>Answer:C</h5>

<h3> 7 Recall that "np.dot(a,b)" performs a matrix multiplication on a and b, whereas "a*b" performs an element-wise multiplication. Consider the two following random arrays "a" and "b": </h3>

```
a = np.random.rand(12288,150) #a.shape = (12288,150)
b = np.random.rand(150,45) #b.shape = (150,45)
c = np.dot(a,b)
```
- A The computation cannot happen because the sizes don't match. It's going to be "Error"!
- B c.shape = (12288, 150)
- C c.shape = (12288, 45)
- D c.shape = (150,150)

```python
import numpy as np

a = np.random.rand(12288,150) #a.shape = (12288,150)
b = np.random.rand(150,45) #b.shape = (150,45)
c = np.dot(a,b)

print(c.shape)
```

```
(12288, 45)
```

<h5>Answer:C</h5>

<h3> 8 Consider the following code snippet: How do you vectorize this? </h3>

```
#a.shape=(3,4)
#b.shape=(4,1)
for i in range(3):
    for j in range(4):
        c[i][j] = a[i][j] + b[j]
```
- A c = a.T + b
- B c = a.T + b.T
- C c = a + b.T
- D c = a + b

<h3> 9 Consider the following code: What will be c? (If you’re not sure, feel free to run this in python to find out).</h3>

```
a = np.random.rand(3,3) #a.shape = (3,3)
b = np.random.rand(3,1) #b.shape = (3,1)
c = a * b
```
- A This will invoke broadcasting, so b is copied three times to become (3,3), and ∗ is an element-wise product so c.shape will be (3, 3)
- B This will invoke broadcasting, so b is copied three times to become (3, 3), and ∗ invokes a matrix multiplication operation of two 3x3 matrices so c.shape will be (3, 3)
- C This will multiply a 3x3 matrix a with a 3x1 vector, thus resulting in a 3x1 vector. That is, c.shape = (3,1).
- D It will lead to an error since you cannot use “*” to operate on these two matrices. You need to instead use np.dot(a,b)

<h3> 10 Consider the following computation graph. What is the output J? </h3>

![neural_network_basics_10](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/neural_network_basics_10.png)

- A J = (c - 1)*(b + a)
- B J = (a - 1) * (b + c)
- C J = a*b + b*c + a*c
- D J = (b - 1) * (c + a)
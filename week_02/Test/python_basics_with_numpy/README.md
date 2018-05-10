## Python Basics With Numpy

<h3> 0 Exercise: Set test to "Hello World" in the cell below to print "Hello World" and run the two cells below. </h3>

```
### START CODE HERE ### (≈ 1 line of code)
test = None
### END CODE HERE ###
print ("test: " + test)
```
```
Expected output: test: Hello World
```
```python
### START CODE HERE ### (≈ 1 line of code)
test = "Hello World"
### END CODE HERE ###
print ("test: " + test)
```
```
test: Hello World
```
<h3> 1 Building basic functions with numpy </h3>

Numpy is the main package for scientific computing in Python. It is maintained by a large community (www.numpy.org). In this exercise you will learn several key numpy functions such as np.exp, np.log, and np.reshape. You will need to know how to use these functions for future assignments.

<h4> 1.1 - sigmoid function, np.exp() </h4>

<h5>Exercise:</h5>
Before using np.exp(), you will use math.exp() to implement the sigmoid function. You will then see why np.exp() is preferable to math.exp().
Build a function that returns the sigmoid of a real number x. Use math.exp(x) for the exponential function.

<h5>Reminder:</h5>
sigmoid(x)=1/(1+e<sup>-x</sup>)  is sometimes also known as the logistic function. It is a non-linear function used not only in Machine Learning (Logistic Regression), but also in Deep Learning.

![python_basics_with_numpy_1_1](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/python_basics_with_numpy_1_1.png)

To refer to a function belonging to a specific package you could call it using package_name.function(). Run the code below to see an example with math.exp().

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

```
# GRADED FUNCTION: basic_sigmoid
​
import math
​
def basic_sigmoid(x):
    """
    Compute sigmoid of x.
​
    Arguments:
    x -- A scalar
​
    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = None
    ### END CODE HERE ###
    
    return s

basic_sigmoid(3)
```
```
Expected Output:basic_sigmoid(3)	0.9525741268224334
```

Actually, we rarely use the "math" library in deep learning because the inputs of the functions are real numbers. 
In deep learning we mostly use matrices and vectors. This is why numpy is more useful.

```
### One reason why we use "numpy" instead of "math" in Deep Learning ###
x = [1, 2, 3]
basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.
```

In fact, if  x=(x<sub>1</sub>,x<sub>2</sub>,...,x<sub>n</sub>) is a row vector then np.exp(x) will apply the exponential function to every element of x. 
The output will thus be: np.exp(x)=(e<sup>x<sub>1</sub></sup>,e<sup>x<sub>2</sub></sup>,...,e<sup>x<sub>n</sub></sup>)

```python
import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))
```

Furthermore, if x is a vector, then a Python operation such as  s=x+3 or s=1/x  will output s as a vector of the same size as x.

```
# example of vector operation
x = np.array([1, 2, 3])
print (x + 3)
```

Any time you need more info on a numpy function, we encourage you to look at the official documentation.

You can also create a new cell in the notebook and write np.exp? (for example) to get quick access to the documentation.

<h5>Exercise:</h5> 

Implement the sigmoid function using numpy.

<h5>Instructions:</h5> 

x could now be either a real number, a vector, or a matrix. The data structures we use in numpy to represent these shapes (vectors, matrices...) are called numpy arrays. You don't need to know more for now.

```
# GRADED FUNCTION: sigmoid
​
import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()
​
def sigmoid(x):
    """
    Compute the sigmoid of x
​
    Arguments:
    x -- A scalar or numpy array of any size
​
    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = None
    ### END CODE HERE ###
    
    return s
    
x = np.array([1, 2, 3])
sigmoid(x)
```
```
Expected Output:sigmoid([1,2,3])	array([ 0.73105858, 0.88079708, 0.95257413])
```

```python
# GRADED FUNCTION: basic_sigmoid

import math
import numpy as np


def basic_sigmoid(x):
    """
    Compute sigmoid of x.
​
    Arguments:
    x -- A scalar
​
    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + math.exp(-x))
    ### END CODE HERE ###

    return s


print('basic_sigmoid(3):', basic_sigmoid(3))

# example of vector operation
x = np.array([1, 2, 3])
print('x:', x)
print('x + 3:', x + 3)

# example of np.exp
x = np.array([1, 2, 3])
print('np.exp(x):', np.exp(x))  # result is (exp(1), exp(2), exp(3))


# GRADED FUNCTION: sigmoid

def sigmoid(x):
    """
    Compute the sigmoid of x
​
    Arguments:
    x -- A scalar or numpy array of any size
​
    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-x))
    ### END CODE HERE ###

    return s


x = np.array([1, 2, 3])
print('sigmoid(x):', sigmoid(x))
```
```
basic_sigmoid(3): 0.9525741268224334
x: [1 2 3]
x + 3: [4 5 6]
np.exp(x): [  2.71828183   7.3890561   20.08553692]
sigmoid(x): [ 0.73105858  0.88079708  0.95257413]
```
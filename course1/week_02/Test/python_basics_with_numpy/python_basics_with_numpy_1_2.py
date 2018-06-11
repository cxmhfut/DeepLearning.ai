from course1.week_02.Test.python_basics_with_numpy.python_basics_with_numpy_1_1 import sigmoid
import numpy as np


# GRADED FUNCTION: sigmoid_derivative
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array
​
    Return:
    ds -- Your computed gradient.
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    s = sigmoid(x)
    ds = s * (1 - s)
    ### END CODE HERE ###

    return ds


x = np.array([1, 2, 3])
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

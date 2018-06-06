import math
import numpy as np


# GRADED FUNCTION: basic_sigmoid
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


if __name__ == '__main__':
    print('basic_sigmoid(3):', basic_sigmoid(3))

    # example of vector operation
    x = np.array([1, 2, 3])
    print('x:', x)
    print('x + 3:', x + 3)

    # example of np.exp
    x = np.array([1, 2, 3])
    print('np.exp(x):', np.exp(x))  # result is (exp(1), exp(2), exp(3))

    x = np.array([1, 2, 3])
    print('sigmoid(x):', sigmoid(x))

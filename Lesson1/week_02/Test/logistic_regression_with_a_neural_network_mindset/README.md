## Logistic Regression with a Neural Network mindset

Welcome to your first (required) programming assignment! 
You will build a logistic regression classifier to recognize cats. 
This assignment will step you through how to do this with a Neural Network mindset, 
and so will also hone your intuitions about deep learning.

<h5>Instructions:</h5>
- Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.

<h5>You will learn to:</h5>
- Build the general architecture of a learning algorithm, including:
>* Initializing parameters
>* Calculating the cost function and its gradient
>* Using an optimization algorithm (gradient descent)
- Gather all three functions above into a main model function, in the right order.

<h3> 1 - Packages </h3>

First, let's run the cell below to import all the packages that you will need during this assignment.

- numpy is the fundamental package for scientific computing with Python.
- h5py is a common package to interact with a dataset that is stored on an H5 file.
- matplotlib is a famous library to plot graphs in Python.
- PIL and scipy are used here to test your model with your own picture at the end.

```
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline
```

<h3> 2 - Overview of the Problem set </h3>

Problem Statement: You are given a dataset ("data.h5") containing:

- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

Let's get more familiar with the dataset. Load the data by running the following code.

```
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```

We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).

Each line of your train_set_x_orig and test_set_x_orig is an array representing an image. You can visualize an example by running the following code. Feel free also to change the index value and re-run to see other images.

```
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
```

Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. 
If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs.

<h5>Exercise:</h5> 
Find the values for:

- m_train (number of training examples)
- m_test (number of test examples)
- num_px (= height = width of a training image)

Remember that train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, you can access m_train by writing train_set_x_orig.shape[0].

```
### START CODE HERE ### (≈ 3 lines of code)
m_train = None
m_test = None
num_px = None
### END CODE HERE ###
​
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
```
```
Expected Output for m_train, m_test and num_px:
m_train	209
m_test	50
num_px	64
```

For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px  ∗∗  num_px  ∗∗  3, 1). 
After this, our training (and test) dataset is a numpy-array where each column represents a flattened image. There should be m_train (respectively m_test) columns.

<h5>Exercise:</h5> 
Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num_px  ∗∗  num_px  ∗∗  3, 1).

A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:

X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

```
# Reshape the training and test examples
​
### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = None
test_set_x_flatten = None
### END CODE HERE ###
​
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
```
```
Expected Output:
train_set_x_flatten shape	(12288, 209)
train_set_y shape	(1, 209)
test_set_x_flatten shape	(12288, 50)
test_set_y shape	(1, 50)
sanity check after reshaping	[17 31 56 22 33]
```
To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.

One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).

Let's standardize our dataset.
```
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```
What you need to remember:

Common steps for pre-processing a new dataset are:

- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
- "Standardize" the data

```python
import numpy as np
import matplotlib.pyplot as plt
from week_02.Test.logistic_regression_with_a_neural_network_mindset.lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
plt.show()
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' picture.")

### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
```

![overview_of_the_problem_set_02](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/overview_of_the_problem_set_02.png)

```
y = [1], it's a 'cat' picture.
Number of training examples: m_train = 209
Number of testing examples: m_test = 50
Height/Width of each image: num_px = 64
Each image is of size: (64, 64, 3)
train_set_x shape: (209, 64, 64, 3)
train_set_y shape: (1, 209)
test_set_x shape: (50, 64, 64, 3)
test_set_y shape: (1, 50)
train_set_x_flatten shape: (12288, 209)
train_set_y shape: (1, 209)
test_set_x_flatten shape: (12288, 50)
test_set_y shape: (1, 50)
sanity check after reshaping: [17 31 56 22 33]
```
<h3> 3 - General Architecture of the learning algorithm </h3>

It's time to design a simple algorithm to distinguish cat images from non-cat images.

You will build a Logistic Regression, using a Neural Network mindset. 
The following Figure explains why Logistic Regression is actually a very simple Neural Network!

![logistic_regression_with_a_neural_network_mindset](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/logistic_regression_with_a_neural_network_mindset.png)

<h5> Mathematical expression of the algorithm: </h5>

For one example  x(i)x(i) :

- z<sup>(i)</sup> = w<sup>T</sup>x<sup>(i)</sup> + b
- ŷ<sup>(i)</sup> = a<sup>(i)</sup> = sigmoid(z<sup>(i)</sup>)
- L(a<sup>(i)</sup>,y<sup>(i)</sup>) = -y<sup>(i)</sup>log(a<sup>(i)</sup>) - (1-y)log(1-a<sup>(i)</sup>)

The cost is then computed by summing over all training examples:

J=(1/m)∑<sub>i 1:m</sub>L(a<sup>(i)</sup>,y<sup>(i)</sup>)

Key steps: In this exercise, you will carry out the following steps:

- Initialize the parameters of the model
- Learn the parameters for the model by minimizing the cost  
- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude

<h3> 4 - Building the parts of our algorithm </h3>

The main steps for building a Neural Network are:

- Define the model structure (such as number of input features)
- Initialize the model's parameters
- Loop:
>* Calculate current loss (forward propagation)
>* Calculate current gradient (backward propagation)
>* Update parameters (gradient descent)

You often build 1-3 separately and integrate them into one function we call model().

<h4> 4.1 - Helper functions </h4>

<h5>Exercise:</h5>
Using your code from "Python Basics", implement sigmoid(). 
As you've seen in the figure above, you need to compute  sigmoid(w<sup>T</sup>x+b)=1/(1+e−(w<sup>T</sup>x+b))  to make predictions. Use np.exp().

```
# GRADED FUNCTION: sigmoid
​
def sigmoid(z):
    """
    Compute the sigmoid of z
​
    Arguments:
    z -- A scalar or numpy array of any size.
​
    Return:
    s -- sigmoid(z)
    """
​
    ### START CODE HERE ### (≈ 1 line of code)
    s = None
    ### END CODE HERE ###
    
    return s
    
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
```
```
Expected Output:sigmoid([0, 2])	[ 0.5 0.88079708]
```

<h4> 4.2 - Initializing parameters </h4>

Exercise: Implement parameter initialization in the cell below. 
You have to initialize w as a vector of zeros. 
If you don't know what numpy function to use, look up np.zeros() in the Numpy library's documentation.

```

# GRADED FUNCTION: initialize_with_zeros
​
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = None
    b = None
    ### END CODE HERE ###
​
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
    
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
```
```
Expected Output:
w	[[ 0.] [ 0.]]
b	0
```
For image inputs, w will be of shape (num_px  ××  num_px  ××  3, 1).

<h4> 4.3 - Forward and Backward propagation </h4>

Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.

<h5>Exercise:</h5>

Implement a function propagate() that computes the cost function and its gradient.

<h5>Hints:</h5>

Forward Propagation:
- You get X
- You compute  A=σ(w<sup>T</sup>X+b)=(a<sup>(0)</sup>,a<sup>(1)</sup>,...,a<sup>(m-1)</sup>,a<sup>(m)</sup>)A=σ(w<sup>T</sup>X+b)=(a<sup>(0)</sup>,a<sup>(1)</sup>,...,a<sup>(m-1)</sup>,a<sup>(m)</sup>) 
- You calculate the cost function:  J=(−1/m)∑<sub>i 1:m</sub>y<sup>(i)</sup>log(a<sup>(i)</sup>) + (1-y)log(1-a<sup>(i)</sup>)

Here are the two formulas you will be using:

- ∂J/∂w=(1/m)X(A−Y)<sup>T</sup>
- ∂J/∂b=(1/m)∑<sub>i 1:m</sub>(a<sup>(i)</sup>−y<sup>(i)</sup>)

```
# GRADED FUNCTION: propagate
​
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
​
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
​
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = None                                     # compute activation
    cost = None                                  # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = None
    db = None
    ### END CODE HERE ###
​
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
```

```
Expected Output:
dw	[[ 0.99993216] [ 1.99980262]]
db	0.499935230625
cost	6.000064773192205
```

<h5>Optimization</h5>
You have initialized your parameters.
You are also able to compute a cost function and its gradient.
Now, you want to update the parameters using gradient descent.

<h5>Exercise:</h5> 
Write down the optimization function. The goal is to learn w and b y minimizing the cost function  JJ . 
For a parameter θ , the update rule is θ=θ−α dθ , where α is the learning rate.

```
# GRADED FUNCTION: optimize
​
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = None
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = None
        b = None
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
​
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
```
```
Expected Output:
w	[[ 0.1124579 ] [ 0.23106775]]
b	1.55930492484
dw	[[ 0.90158428] [ 1.76250842]]
db	0.430462071679
```

<h5>Exercise:</h5>
The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the predict() function. There is two steps to computing predictions:

- Calculate  Ŷ =A=σ(wTX+b)Y^=A=σ(wTX+b) 
- Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector Y_prediction. If you wish, you can use an if/else statement in a for loop (though there is also a way to vectorize this).

```
# GRADED FUNCTION: predict
​
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = None
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        pass
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
    
print ("predictions = " + str(predict(w, b, X)))
```
```
Expected Output:
predictions	[[ 1. 1.]]
```

What to remember: You've implemented several functions that:

- Initialize (w,b)
- Optimize the loss iteratively to learn parameters (w,b):
>* computing the cost and its gradient
>* updating the parameters using gradient descent
- Use the learned (w,b) to predict the labels for a given set of examples

```python
import numpy as np


# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z
​
    Arguments:
    z -- A scalar or numpy array of any size.
​
    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s


# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# GRADED FUNCTION: propagate


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
​
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
​
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    ### END CODE HERE ###

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w,b,X,Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
        ### END CODE HERE ###

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

if __name__ == '__main__':
    print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))

    dim = 2
    w, b = initialize_with_zeros(dim)
    print("w = " + str(w))
    print("b = " + str(b))

    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    grads, cost = propagate(w, b, X, Y)
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("cost = " + str(cost))

    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print(costs)

    print("predictions = " + str(predict(w, b, X)))
```
```
sigmoid([0, 2]) = [ 0.5         0.88079708]
w = [[ 0.]
     [ 0.]]
b = 0
dw = [[ 0.99993216]
      [ 1.99980262]]
db = 0.499935230625
cost = 6.00006477319
w = [[ 0.1124579 ]
     [ 0.23106775]]
b = 1.55930492484
dw = [[ 0.90158428]
      [ 1.76250842]]
db = 0.430462071679
[6.0000647731922054]
predictions = [[ 1.  1.]]
```
<h3> 5 - Merge all functions into a model </h3>

You will now see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.

<h5>Exercise:</h5>
Implement the model function. Use the following notation:

- Y_prediction for your predictions on the test set
- Y_prediction_train for your predictions on the train set
- w, costs, grads for the outputs of optimize()

```
# GRADED FUNCTION: model
​
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = None
​
    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = None
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = None
    Y_prediction_train = None
​
    ### END CODE HERE ###
​
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
​
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
```
Run the following cell to train your model.
```
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

Expected Output:
Train Accuracy	99.04306220095694 %
Test Accuracy	70.0 %
```

```python
import numpy as np
import matplotlib.pyplot as plt
from week_02.Test.logistic_regression_with_a_neural_network_mindset.lr_utils import load_dataset
from week_02.Test.logistic_regression_with_a_neural_network_mindset.load_data import load_data
from week_02.Test.logistic_regression_with_a_neural_network_mindset.model import model

# Loading the data (cat/non-cat)
train_set_x_orig, _, test_set_x_orig, _, classes = load_dataset()
train_set_x, train_set_y, test_set_x, test_set_y = load_data()
num_iterations = 2000
learning_rate = 0.005
num_px = train_set_x_orig.shape[1]


def train(print_cost=False):
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate, print_cost)
    return d


if __name__ == '__main__':
    d = train(True)

    # Example of a picture that was wrongly classified.
    index = 1
    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    plt.show()
    print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
        int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")

    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
```
```
Cost after iteration 0: 0.693147
Cost after iteration 100: 0.584508
Cost after iteration 200: 0.466949
Cost after iteration 300: 0.376007
Cost after iteration 400: 0.331463
Cost after iteration 500: 0.303273
Cost after iteration 600: 0.279880
Cost after iteration 700: 0.260042
Cost after iteration 800: 0.242941
Cost after iteration 900: 0.228004
Cost after iteration 1000: 0.214820
Cost after iteration 1100: 0.203078
Cost after iteration 1200: 0.192544
Cost after iteration 1300: 0.183033
Cost after iteration 1400: 0.174399
Cost after iteration 1500: 0.166521
Cost after iteration 1600: 0.159305
Cost after iteration 1700: 0.152667
Cost after iteration 1800: 0.146542
Cost after iteration 1900: 0.140872
train accuracy: 99.04306220095694 %
test accuracy: 70.0 %
y = 1, you predicted that it is a "cat" picture.
```
Comment: Training accuracy is close to 100%. This is a good sanity check: your model is working and has high enough capacity to fit the training data. Test error is 68%. It is actually not bad for this simple model, given the small dataset we used and that logistic regression is a linear classifier. But no worries, you'll build an even better classifier next week!

Also, you see that the model is clearly overfitting the training data. Later in this specialization you will learn how to reduce overfitting, for example by using regularization. Using the code below (and changing the index variable) you can look at predictions on pictures of the test set.

```
# Example of a picture that was wrongly classified.
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
```

![merge_all_function_into_a_model_05_1](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/merge_all_function_into_a_model_05_1.png)

Let's also plot the cost function and the gradients.

```
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
```

![merge_all_function_into_a_model_05_2](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/merge_all_function_into_a_model_05_2.png)

Interpretation: You can see the cost decreasing. 
It shows that the parameters are being learned. 
However, you see that you could train the model even more on the training set. 
Try to increase the number of iterations in the cell above and rerun the cells. 
You might see that the training set accuracy goes up, but the test set accuracy goes down. 
This is called overfitting.

<h3> 6 - Further analysis (optional/ungraded exercise) </h3>

Congratulations on building your first image classification model. Let's analyze it further, and examine possible choices for the learning rate α.

Choice of learning rate
<h5>Reminder:</h5>
In order for Gradient Descent to work you must choose the learning rate wisely. 
The learning rate  αα  determines how rapidly we update the parameters. 
If the learning rate is too large we may "overshoot" the optimal value. 
Similarly, if it is too small we will need too many iterations to converge to the best values. 
That's why it is crucial to use a well-tuned learning rate.

Let's compare the learning curve of our model with several choices of learning rates. Run the cell below. This should take about 1 minute. Feel free also to try different values than the three we have initialized the learning_rates variable to contain, and see what happens.

```
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')
​
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
​
plt.ylabel('cost')
plt.xlabel('iterations')
​
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```
```python
import numpy as np
import matplotlib.pyplot as plt
from week_02.Test.logistic_regression_with_a_neural_network_mindset.load_data import load_data
from week_02.Test.logistic_regression_with_a_neural_network_mindset.model import model

train_set_x, train_set_y, test_set_x, test_set_y = load_data()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                           print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```
```
learning rate is: 0.01
train accuracy: 99.52153110047847 %
test accuracy: 68.0 %

-------------------------------------------------------

learning rate is: 0.001
train accuracy: 88.99521531100478 %
test accuracy: 64.0 %

-------------------------------------------------------

learning rate is: 0.0001
train accuracy: 68.42105263157895 %
test accuracy: 36.0 %

-------------------------------------------------------
```
![further_analysis_06](https://github.com/cxmhfut/DeepLearning.ai/blob/master/images/further_analysis_06.png)

Interpretation:

- Different learning rates give different costs and thus different predictions results.
- If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).
- A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
- In deep learning, we usually recommend that you:
>* hoose the learning rate that better minimizes the cost function.
>* If your model overfits, use other techniques to reduce overfitting. (We'll talk about this in later videos.)

<h3> 7 - Test with your own image (optional/ungraded exercise) </h3>

Congratulations on finishing this assignment. You can use your own image and see the output of your model. To do that:

- 1 Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
- 2 Add your image to this Jupyter Notebook's directory, in the "images" folder
- 3 Change your image's name in the following code
- 4 Run the code and check if the algorithm is right (1 = cat, 0 = non-cat)!

```
## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "my_image.jpg"   # change this to the name of your image file 
## END CODE HERE ##
​
# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)
​
plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```
What to remember from this assignment:

- 1 Preprocessing the dataset is important.
- 2 You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
- 3 Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. You will see more examples of this later in this course!

Finally, if you'd like, we invite you to try different things on this Notebook. Make sure you submit before trying anything. Once you submit, things you can play with include:

- Play with the learning rate and the number of iterations
- Try different initialization methods and compare the results
- Test other preprocessings (center the data, or divide each row by its standard deviation)
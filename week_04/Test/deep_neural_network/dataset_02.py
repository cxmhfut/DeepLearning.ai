import matplotlib.pyplot as plt
from week_04.Test.deep_neural_network.lr_utils import load_dataset

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Example of picture
index = 10
plt.imshow(train_x_orig[index])
plt.show()
print("y=" + str(train_y[0, index]) + ". It is a " + classes[train_y[0, index]].decode('utf-8') + " picture.")

# Explore your dataset

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and testing examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
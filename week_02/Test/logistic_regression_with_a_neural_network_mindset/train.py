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
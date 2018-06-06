from Lesson1.week_02.Test.logistic_regression_with_a_neural_network_mindset.lr_utils import load_dataset


def load_data():
    """load normalized data

    origin data:
    train_set_x_orig    shape (m_train, num_px, num_px, 3)
    train_set_y         shape (1, m_train)
    test_set_x_orig     shape (m_test, num_px, num_px, 3)
    test_set_y          shape (1, m_test)

    """
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    return train_set_x, train_set_y, test_set_x, test_set_y

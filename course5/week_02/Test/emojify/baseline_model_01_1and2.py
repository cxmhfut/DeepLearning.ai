from course5.week_02.Test.emojify.emo_utils import *


def load_data(print_info=False):
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')

    maxLen = len(max(X_train, key=len).split())

    index = 1
    if print_info: print(X_train[index], label_to_emoji(Y_train[index]))

    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)

    index = 50
    if print_info: print(Y_train[index], "is converted into one hot", Y_oh_train[index])

    return X_train,Y_train,X_test,Y_test,Y_oh_train,Y_oh_test

if __name__ == '__main__':
    load_data(True)
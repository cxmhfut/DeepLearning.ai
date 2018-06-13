from course5.week_02.Test.emojify.emojifier_V1_01_1and2 import load_data
from course5.week_02.Test.emojify.emojifier_V1_01_3 import model
from course5.week_02.Test.emojify.emo_utils import *

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
X_train, Y_train, X_test, Y_test, Y_oh_train, Y_oh_test, maxLen = load_data()

pred, W, b = model(X_train, Y_train, word_to_vec_map)

print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

X_my_sentences = np.array(
    ["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])

pred = predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)

print(Y_test.shape)
print('           ' + label_to_emoji(0) + '    ' + label_to_emoji(1) + '    ' + label_to_emoji(
    2) + '    ' + label_to_emoji(3) + '   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56, ), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)

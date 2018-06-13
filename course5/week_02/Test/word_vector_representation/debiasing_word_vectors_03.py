from course5.week_02.Test.word_vector_representation.w2v_utils import *
from course5.week_02.Test.word_vector_representation.cosine_similarity_01 import cosine_similarity


def debiasing_test_case():
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    g = word_to_vec_map['woman'] - word_to_vec_map['man']
    print(g)

    print('List of names and their similarities with constructed vector:')

    # girls and boys name
    name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

    for w in name_list:
        print(w, cosine_similarity(word_to_vec_map[w], g))

    print('Other words and their similarities:')
    word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior', 'doctor', 'tree', 'receptionist',
                 'technology', 'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
    for w in word_list:
        print(w, cosine_similarity(word_to_vec_map[w], g))


def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    ### START CODE HERE ###
    # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
    e = word_to_vec_map[word]

    # Compute e_biascomponent using the formula give above. (≈ 1 line)
    e_biascomponent = np.multiply(np.dot(e, g) / np.linalg.norm(g) ** 2, g)

    # Neutralize e by substracting e_biascomponent from it
    # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
    e_debiased = e - e_biascomponent
    ### END CODE HERE ###

    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    ### START CODE HERE ###
    # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
    mu = (e_w1 + e_w2) / 2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
    mu_B = np.dot(mu, bias_axis) / (np.linalg.norm(bias_axis) ** 2) * bias_axis
    mu_orth = mu - mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
    e_w1B = np.dot(e_w1, bias_axis) / (np.linalg.norm(bias_axis) ** 2) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / (np.linalg.norm(bias_axis) ** 2) * bias_axis

    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
    corrected_e_w1B = np.sqrt(np.abs(1 - np.linalg.norm(mu_orth) ** 2)) * (e_w1B - mu_B) / np.abs(
        (e_w1 - mu_orth) - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1 - np.linalg.norm(mu_orth) ** 2)) * (e_w2B - mu_B) / np.abs(
        (e_w2 - mu_orth) - mu_B)

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    ### END CODE HERE ###

    return e1, e2

if __name__ == '__main__':
    # debiasing_test_case()
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    g = word_to_vec_map['woman'] - word_to_vec_map['man']
    e = "receptionist"
    print("cosine similarity between " + e + " and g, before neutralizing: ",
          cosine_similarity(word_to_vec_map["receptionist"], g))

    e_debiased = neutralize("receptionist", g, word_to_vec_map)
    print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))

    print("cosine similarities before equalizing:")
    print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
    print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
    print()
    e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
    print("cosine similarities after equalizing:")
    print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
    print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))

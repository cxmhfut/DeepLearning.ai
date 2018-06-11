def load_data(print_info=False):
    data = open('dinos.txt', 'r').read()
    data = data.lower()
    chars = set(list(data))
    data_size, vocab_size = len(data), len(chars)
    if print_info:
        print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    if print_info:
        print(ix_to_char)

    return data, vocab_size, char_to_ix, ix_to_char


if __name__ == '__main__':
    load_data(True)

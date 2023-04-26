# Ref: https://www.youtube.com/watch?v=kCc8FmEb1nY
# -----------------------------------------------------------------------------------
import numpy as np
import torch


def set_random_seed(s):
    """
    Fix the random seed across all modules for reproducibility
    :param s:
    :return:
    """
    torch.manual_seed(s)
    np.random.seed(s)


def get_batch(data_type, b_size, max_c_len, t_data, v_data):
    """

    #note this does not guarentee going over all the lines in the code. Relies on
    being called multiple times to probabilistically cover the whole dataset.
    But the advantage of this apporach is that it can handle variable length context

    :param v_data: pointer to validation data
    :param t_data: pointer to train data
    :param data_type: ['train', 'val']
    :param b_size: batch_size
    :param max_c_len:  max context Length
    :return:
    """
    valid_data_types = ['train', 'val']
    if data_type not in valid_data_types:
        raise Exception("invalid data type {}. Can only be {}".format(data_type, valid_data_types))

    use_data = t_data if data_type == 'train' else v_data
    # choose random integers between 1 and max_length to generate context
    max_len = len(use_data)
    rand_int_sequence = np.random.randint(1, max_len - max_c_len, size=b_size)

    x = []
    y = []
    for rand_int in rand_int_sequence:
        x.append(use_data[rand_int: rand_int + max_c_len])
        y.append(use_data[rand_int + 1: rand_int + max_c_len + 1])
    x = torch.stack(x, 0)
    y = torch.stack(y, 0)

    # note y is also a sequence, not a single label. This is done for handling variable context length in training
    return x, y


def main(raw_data_strings):
    """

    :param raw_data_strings: training data
    :return:
    """
    # ----------------------------------------------------------------------------
    # Create Vocab
    # ----------------------------------------------------------------------------
    # Get list of all unique characters in the dataset
    unique_char = set(raw_data_strings)
    unique_char = sorted(unique_char)
    vocab_size = len(unique_char)

    print("Number of unique characters {}:\n{}".format(vocab_size, unique_char))

    # ----------------------------------------------------------------------------
    # Tokenize the text - Convert input_data to integer repr
    # character level vocab
    # ----------------------------------------------------------------------------
    ctoi = {c: i for i, c in enumerate(unique_char)}  # dictionary for mapping character to i
    itoc = {i: c for i, c in enumerate(unique_char)}  # dictionary for mapping integer to character

    encode = lambda string: [ctoi[c] for c in string]  # encode a whole string, not just a character
    decode = lambda int_string: ''.join([itoc[i] for i in int_string])  # decode a whole string, not just a character

    # # Test encoder/decoder
    # test_string = 'Salman is the best'
    # # print("'{}' encoded as :  {}".format(test_string, encode(test_string)))
    # # print("Decoded test string is '{}'".format(decode(encode(test_string))))
    #
    # # Other more standard encoder/decodes
    # import tiktoken
    # encoder = tiktoken.get_encoding('gpt2')
    # y= encoder.encode(test_string)

    input_data = encode(raw_data_strings)  # converted to integers
    input_data = torch.Tensor(input_data)

    # ----------------------------------------------------------------------------
    # Split Data to Train/Validation Set
    # ----------------------------------------------------------------------------
    train_val_split = 0.9
    n = int(len(input_data)*train_val_split)
    train_data = input_data[:n]
    val_data = input_data[n:]

    # ----------------------------------------------------------------------------
    # Data loading and context size
    # ----------------------------------------------------------------------------
    context_size = 8
    batch_size = 32

    # # Try out a simple example
    # x = train_data[:context_size]
    # y = train_data[1: context_size + 1]
    # # Check that the input and the labels make sense
    # for i in range(context_size):
    #     print("For input {}, label is {}".format(x[:i+1], y[i]))
    # note  the value of wy and how it is used. This is used for training with different sequence lengths.

    # Actual Data loader
    x_in, y_label = get_batch('train', batch_size, context_size, train_data, val_data)
    for b in range(batch_size):
        print(x_in[b], y_label[b])
    #
    # for b in range(batch_size):
    #     for t in range(context_size):
    #         print("{} input {}, label={}".format(b, x_in[b, :t+1], y_label[b, t]))

    # ----------------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------------


if __name__ == "__main__":
    set_random_seed(9)

    data_file = './data/input.txt'
    with open(data_file, 'r') as handle:
        data = handle.read()

    print("Number of characters in data {}".format(len(data)))

    main(data)

    import pdb
    pdb.set_trace()

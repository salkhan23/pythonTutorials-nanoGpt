# Ref: https://www.youtube.com/watch?v=kCc8FmEb1nY
# -----------------------------------------------------------------------------------
import numpy as np
import torch


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


    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    data_file = './data/input.txt'

    with open(data_file, 'r') as handle:
        data = handle.read()

    print("Number of characters in data {}".format(len(data)))

    main(data)

    import pdb
    pdb.set_trace()

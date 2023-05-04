# Ref: https://www.youtube.com/watch?v=kCc8FmEb1nY
# -----------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_random_seed(s):
    """
    Fix the random seed across all modules for reproducibility
    :param s:
    :return:
    """
    torch.manual_seed(s)
    np.random.seed(s)


def get_batch(data_type, b_size, max_c_len, t_data, v_data, device):
    """

    #note this does not guarantee going over all the lines in the code. Relies on
    being called multiple times to probabilistically cover the whole dataset.
    But the advantage of this approach is that it can handle variable length context

    :param device:
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

    x = x.to(device)
    y = y.to(device)

    # note y is also a sequence, not a single label. This is done for handling variable context length in training
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size_1):
        """
        In the bigram model we use only the current character/word to predict what the next character/word
        should be. The model calculates the probability of the next character given the current character/word.

        """
        super().__init__()
        self.vocab_size = vocab_size_1
        # create and embedding table of dimension [Vocab size X Vocab size]
        # Each embedding is of size vocab_size and there are vocab_size embeddings
        self.embedded_token_table = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.vocab_size)

    def forward(self, input_idx_arr, tgt_labels_arr=None):
        """ Given an [B,T] matrix of indexes and [B,T] array of labels, get embeddings and compare them with labels

        """
        # nn.Embeddings returns embeddings for each input
        logits1 = self.embedded_token_table(input_idx_arr)  # [B,T] --> [B, T, CH], an embedding for each [B, T]
        b, t, ch = logits1.shape

        # Compare predictions with Labels using negative cross entropy
        # Reshape the input and the label to shapes expected by pytorch cross_entropy function
        # [B, ]
        logits1_flatten = logits1.view(b*t, ch)

        if tgt_labels_arr is not None:
            tgt_labels_arr = tgt_labels_arr.view(b * t)
            loss1 = F.cross_entropy(logits1_flatten, tgt_labels_arr)
        else:
            loss1 = 0

        return logits1, loss1

    def generate(self, input_idx_arr, n_new_tokens):
        """
        Given an input_idx_arr of size [B, T(Time)] generate n_new tokens for each B(batch)
        :param input_idx_arr:
        :param n_new_tokens:
        :return:  [1, n_new_token_size]
        """
        for n_idx in range(n_new_tokens):
            logits2, loss2 = self(input_idx_arr)  # somehow calls the forward function. Prob a feature of nn.Module
            logits2 = logits2[:, -1, :]
            # Get the last time, set of logits
            probs = F.softmax(logits2, dim=1)

            # chooses 1 (num_samples) of probs, based on probabilies specifed in probs
            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            input_idx_arr = torch.cat((input_idx_arr, idx_next), dim=1)  # on time axes

        return input_idx_arr


@torch.no_grad
def evaluate_loss(model, eval_iters, b_size, max_c_len, t_data, v_data, device):
    model.eval()
    loss = {}

    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, b_size, max_c_len, t_data, v_data, device)
            logits2, loss2 = model(xb, yb)
            losses[k] = loss2.item()
        loss[split] = losses.mean()

    model.train()
    return loss


def main(raw_data_strings):
    """

    :param raw_data_strings: training data
    :return:
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_val_split = 0.9
    context_size = 8
    batch_size = 256
    n_iters = 10000
    n_eval_iters = 200

    # ----------------------------------------------------------------------------
    # Create Vocab
    # ----------------------------------------------------------------------------
    unique_char = set(raw_data_strings)  # Get list of all unique characters in the dataset
    unique_char = sorted(unique_char)
    vocab_size = len(unique_char)

    print("Unique characters (Total= {}):\n{}".format(vocab_size, unique_char))

    # ----------------------------------------------------------------------------
    # Tokenize the text - Convert input_data to integer representation
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
    # # Other more 'standard' tokenizers
    # import tiktoken # uses by GPT.
    # encoder = tiktoken.get_encoding('gpt2')  # Bit-pair encoding
    # y = encoder.encode(test_string)

    input_data = encode(raw_data_strings)  # converted to integers
    input_data = torch.tensor(input_data, dtype=torch.long)

    # ----------------------------------------------------------------------------
    # Split Data to Train/Validation Set
    # ----------------------------------------------------------------------------
    n = int(len(input_data)*train_val_split)
    train_data = input_data[:n]
    val_data = input_data[n:]

    # ----------------------------------------------------------------------------
    # Data loading and context size
    # ----------------------------------------------------------------------------
    # # Try out a simple example
    # x = train_data[:context_size]
    # y = train_data[1: context_size + 1]
    # # Check that the input and the labels make sense
    # for i in range(context_size):
    #     print("For input {}, label is {}".format(x[:i+1], y[i]))
    # note  the value of wy and how it is used. This is used for training with different sequence lengths.

    # Actual Data loader
    x_in, y_label = get_batch('train', batch_size, context_size, train_data, val_data, device)
    # x_in = [B, T], y_in = [B,T]; B = Batch, T = Time

    # for b in range(batch_size):
    #     print(x_in[b], y_label[b])
    # #
    # # for b in range(batch_size):
    # #     for t in range(context_size):
    # #         print("{} input {}, label={}".format(b, x_in[b, :t+1], y_label[b, t]))

    # ----------------------------------------------------------------------------
    # Model - Simple Bigram Model
    # Predicts the probability of the next work, given only the previous word.
    # ----------------------------------------------------------------------------
    model1 = BigramLanguageModel(vocab_size)
    model1.to(device)

    # logits, loss = model1(x_in, y_label)
    # print("Loss {}".format(loss))

    # Get some predicts
    predicts = model1.generate(torch.zeros((1, 1), dtype=torch.long).to(device), 50)
    output = ''.join(itoc[i.item()] for i in predicts[0])
    print("Generated text {}".format(output))

    # ----------------------------------------------------------------------------
    # Train the Model
    # ----------------------------------------------------------------------------
    print("Starting Training")
    optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)

    model1.train()
    yb_loss = 0
    idx = 0
    for idx in range(n_iters):
        xb, yb = get_batch('train', batch_size, context_size, train_data, val_data,device)

        if idx % 1000 == 0:
            losses_evaluated = \
                evaluate_loss(model1, n_eval_iters, batch_size, context_size, train_data, val_data, device)
            print("{} Train {:0.4f}, Val {:0.4f}".format(idx, losses_evaluated['train'], losses_evaluated['val']))

        yb_logits, yb_loss = model1(xb, yb)
        # print("{}: {:0.4f}".format(idx, yb_loss.item()))

        optimizer.zero_grad()
        yb_loss.backward()
        optimizer.step()

    print("Final Loss: num_iters {}: {:0.4f}".format(idx, yb_loss.item()))

    # ----------------------------------------------------------------------------
    # Test the model
    # ----------------------------------------------------------------------------
    # Get some predicted names
    predicts = model1.generate(torch.zeros((1, 1), dtype=torch.long).to(device), 50)
    output = ''.join(itoc[i.item()] for i in predicts[0])
    print("Generated text after training: {}".format(output))

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    set_random_seed(9)

    data_file = './data/input.txt'
    with open(data_file, 'r') as handle:
        data = handle.read()

    print("Number of characters in data {}".format(len(data)))

    main(data)

    import pdb
    pdb.set_trace()

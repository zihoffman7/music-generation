import numpy as np
import os, re

# Holds each note/chord corresponding to a number
vocab = {}
vectorized = {}
 # Max length of the training vector
GENSIZE = 500
SEQ_LEN = 10
# Max vocabulary that will be learned
VOCAB_THRESHOLD = 2000

def parse_abc(path, gensize, key="c", time_signature="4/4"):
    # Ensure that only abc files are opened
    if path.endswith(".abc"):
        with open(path) as f:
            content = f.readlines()
            # Check if the key is in C major
            if not f"k:{key.lower()}" in "".join(content).lower() and not f"k: {key.lower()}" in "".join(content).lower():
                return False
            # Check if the time signature is 4/4
            if not f"m:{time_signature.lower()}" in "".join(content).lower() and not f"m: {time_signature.lower()}" in "".join(content).lower():
                return False
            # Remove unnecessary abc characters and cut model to not exceed max length
            try:
                return list(filter(lambda a: a != " " and a != "\n" and a != "/", re.split("(\s|\/|\||\n)", "".join(content[[i for i, s in enumerate(content) if "K:" in s][0] + 1:]).replace("\n", "").replace("\\", ""))))[:gensize]
            except IndexError:
                return False

def split_data(data, gensize, seq_len):
    char_len = len(data)

    data_x = []
    data_y = []

    for j in data:
        for i in range(0, char_len - seq_len):
            print(i)
            if (i + seq_len >= len(j)):
                break
            inp = j[i:i + seq_len]
            out = j[i + seq_len]
            try:
                data_x.append([vocab[char] for char in inp])
                data_y.append(vocab[out])
            except KeyError:
                continue
    return data_x, data_y

def load_data(gensize=320, vocab_threshold=5000, seq_len=50, **kwargs):
    global vocab
    # Will hold the processed train and test data
    data = []
    # Iterate through all abc files in song directory
    for file in os.listdir("songs"):
        # Try to extract the valid abcd
        x = parse_abc(f"songs/{os.fsdecode(file)}", gensize, **kwargs)
        if not x:
            continue
        for i in x:
            if len(vocab) + len(x) >= vocab_threshold:
                break
            vocab = {**vocab, **dict((c, i) for i, c in enumerate(x) if len(c))}
        data.append(list(filter(lambda a: len(a), x)))

    x, y = split_data(data, gensize, seq_len)

    # training data is now [patterns, size, 1]
    training_x = np.reshape(x, (len(x), seq_len, 1)) / float(len(vocab.keys()))
    training_y = np.eye(len(vocab))[y]

    print(training_x.shape)
    print(training_y.shape)

    return training_x, training_y

# Split data into train and test
train_x, train_y = load_data(GENSIZE, VOCAB_THRESHOLD, SEQ_LEN)


# Data summary
print(f"Length of training data: {len(train_x)}")

# from music21 import converter
# s = converter.parse('test.abc')
# fp = s.write('midi', fp='text.mid')

import numpy as np
import os, re
from keras.utils import np_utils

# Holds each note/chord corresponding to a number
vocab = {}
vectorized = {}
 # Max length of the training vector
GENSIZE = 500
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
                return list(filter(lambda a: a != " " and a != "\n", re.split("(\s|\/|\||\n)", "".join(content[[i for i, s in enumerate(content) if "K:" in s or "%" in s or "C:" in s or "\n" == s][0] + 1:]).replace("\n", "").replace("\\", ""))))[:gensize]
            except IndexError:
                return False

def load_data(gensize=320, vocab_threshold=5000, **kwargs):
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
            vocab = {**vocab, **dict((c, i) for i, c in enumerate(x))}
        data.append("".join(x))

    char_len = len("".join(data))
    vocab_keys = sorted(list(set(vocab.keys())))

    data_x = []
    data_y = []

    for i in range(0, char_len - gensize):
        if (i + gensize >= len(vocab_keys)):
            break
        inp = vocab_keys[i:i + gensize]
        out = vocab_keys[i + gensize]
        data_x.append([vocab[char] for char in inp])
        data_y.append(vocab[out])
    # training data is now [patterns, size, 1]
    training_x = np.reshape(data_x, (len(data_x), gensize, 1)) / float(len(vocab.keys()))
    training_y = np.eye(len(vocab))[data_y]

    print(training_x)
    print(training_x.shape)
    print("-------")
    print(training_y)
    print(training_y.shape)
    return training_x, training_y

# Split data into train and test
train_x, train_y = load_data(GENSIZE, VOCAB_THRESHOLD)


# Data summary
print(f"Length of training data: {len(train_x)}")

# from music21 import converter
# s = converter.parse('test.abc')
# fp = s.write('midi', fp='text.mid')

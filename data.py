import numpy as np
import os, re

# Holds each note/chord corresponding to a number
vocab = {}
vectorized = {}
 # Max length of the training vector
GENSIZE = 320
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
                return list(filter(lambda a: a != " " and a != "\n", re.split("(\s|\/|\||\n)", "".join(content[[i for i, s in enumerate(content) if "K:" in s or "\n" == s][0] + 1:]).replace("\n", "").replace("\\", ""))))[:gensize]
            except IndexError:
                return False

def load_data(gensize=320, **kwargs):
    # Will hold the processed train and test data
    data = []
    # Iterate through all abc files in song directory
    for file in os.listdir("songs"):
        # Try to extract the valid abc
        x = parse_abc(f"songs/{os.fsdecode(file)}", gensize, **kwargs)
        if not x:
            continue
        # Create empty vector of length gensize
        m = np.zeros([gensize], dtype=np.int32)
        # Iterate through each value of abc
        for i in x:
            # Exit process if too much vocab
            if len(vocab) >= VOCAB_THRESHOLD:
                return data[:round(len(data) * 0.8)], data[round(len(data) * 0.8):]
            # Add the current value to vocab if not there already
            if not i in vocab:
                vocab[i] = len(vocab.keys())
                vectorized[i] = len(vocab.keys()) + 1
            # Add the value's vectorized number to the data vector
            if i in vectorized.keys():
                m[x.index(i)] = int(vectorized[i])
        # Add the song vector to the data
        data.append(m)
    # Split data into training (80%) and test (20%)
    return data[:round(len(data) * 0.8)], data[round(len(data) * 0.8):]

# Split data into train and test
train, test = load_data(GENSIZE)
# Data summary
print(f"Length of train data: {len(train)} songs\nLength of test data: {len(test)} songs\nTotal vocab length: {len(vocab)}")

# from music21 import converter
# s = converter.parse('test.abc')
# fp = s.write('midi', fp='text.mid')

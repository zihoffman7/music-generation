import numpy as np
import os, re

# Holds an element corresponding to a number
vocab = {}
 # Max length of the training vector
GEN_SIZE = 5000
# Length of patterns to be detected
SEQ_LEN = 10
# Max vocabulary that will be learned
VOCAB_THRESHOLD = 2000

def parse_abc(path, gen_size, key="c", time_signature="4/4"):
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
                return list(filter(lambda a: a != " " and a != "\n" and a != "/" and a != "" and a[0] != "%", re.split("(\s|\/|\||\n)", "".join(content[[i for i, s in enumerate(content) if "v:1\n" in s.lower() or "v:1 " in s.lower()][1] + 1:[i for i, s in enumerate(content) if "v:2\n" in s.lower() or "v:2 " in s.lower()][1]]).replace("\n", "").replace("\\", ""))))[:gen_size]
            except IndexError:
                print("oof")
                return False

def split_data(data, gen_size, seq_len):
    char_len = sum([len(x) for x in data])

    data_x = []
    data_y = []

    # Iterate through data
    for d in data:
        # Extract a sequence of seq_len elements as x and the element after as y
        for i in range(0, char_len - seq_len):
            if (i + seq_len >= len(d)):
                break
            inp = d[i:i + seq_len]
            out = d[i + seq_len]
            try:
                data_x.append([vocab[char] for char in inp])
                data_y.append(vocab[out])
            except KeyError:
                try:
                    if data_x[-1] == [vocab[char] for char in inp]:
                        data_x.pop(-1)
                except KeyError:
                    continue
                continue
    return data_x, data_y

def load_data(gen_size=320, vocab_threshold=5000, seq_len=50, **kwargs):
    global vocab
    # Will hold the processed train and test data
    data = []
    # Iterate through all abc files in song directory
    for file in os.listdir("abc"):
        # Try to extract the valid abcd
        s = parse_abc(f"abc/{os.fsdecode(file)}", gen_size, **kwargs)
        if not s:
            continue
        for i in s:
            if len(vocab) + len(s) >= vocab_threshold:
                break
            vocab = {**vocab, **dict((c, i) for i, c in enumerate(s) if len(c))}
        data.append(list(filter(lambda a: len(a), s)))

    x, y = split_data(data, gen_size, seq_len)
    # train x = [number of patterns, size of pattern, 1]
    training_x = np.reshape(x, (len(x), seq_len, 1))

    # Resize train x to be between zero and 1
    training_x = training_x / float(len(vocab))
    # Designate each column for a vocab term.
    # A '1' in column represents its corresponding vocab term
    training_y = np.eye(len(y), len(vocab))[y]

    return training_x, training_y

# Split data into train and test
train_x, train_y = load_data(GEN_SIZE, VOCAB_THRESHOLD, SEQ_LEN)

# Data summary
print(f"Train x shape: {train_x.shape}")
print(f"Train y shape: {train_y.shape}")
print(f"Length of training data: {len(train_x)}")

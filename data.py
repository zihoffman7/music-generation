import numpy as np
import os, re, random

# Holds an element corresponding to a number
vocab = {}
# Length of patterns to be detected
SEQ_LEN = 10
# Max vocabulary that can be learned
VOCAB_THRESHOLD = 2000

def parse_abc(path, key="c", time_signature="4/4"):
    if path.endswith(".abc"):
        with open(path) as f:
            content = f.readlines()
            # Check if the key is in C major
            if not f"k:{key.lower()}" in "".join(content).lower() and not f"k: {key.lower()}" in "".join(content).lower() and not "k:none" in "".join(content).lower():
                return False
            # Check if the time signature is 4/4
            if not f"m:{time_signature.lower()}" in "".join(content).lower() and not f"m: {time_signature.lower()}" in "".join(content).lower():
                return False
            # Remove unnecessary abc characters and cut model to not exceed max length
            try:
                return list(filter(lambda a: a != " " and a != "\n" and a != "/" and a != "" and a[0] != "%", re.split("(\s|\/|\||\n)", "".join(content[[i for i, s in enumerate(content) if "v:1\n" in s.lower() or "v:1 " in s.lower()][1] + 1:[i for i, s in enumerate(content) if "v:2\n" in s.lower() or "v:2 " in s.lower()][1]]).replace("\n", "").replace("\\", ""))))
            except IndexError:
                return False

def split_data(data, seq_len):
    global vocab
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

    # train x = [number of patterns, size of pattern, 1]
    data_x = np.reshape(data_x, (len(data_x), seq_len, 1))
    # Resize train x to be between zero and 1
    data_x = data_x / float(len(vocab))
    # Designate each column for a vocab term.
    # A '1' in column represents its corresponding vocab term
    d_y = np.eye(len(data_y), len(vocab))[data_y]
    return data_x, d_y

def shuffle_data(x, y):
    # Shuffle train and test data
    c = list(zip(x, y))
    random.shuffle(c)
    return zip(*c)

def load_data(vocab_threshold=5000, seq_len=50, **kwargs):
    global vocab
    # Will hold the processed train and test data
    data = []
    # Iterate through all abc files in song directory
    for file in os.listdir("abc"):
        # Try to extract the valid abcd
        s = parse_abc(f"abc/{os.fsdecode(file)}", **kwargs)
        if not s:
            continue
        for i in s:
            if len(vocab) + len(s) >= vocab_threshold:
                break
            vocab = {**vocab, **dict((c, len(vocab)) for c in s if len(c) and c not in vocab.keys())}
        data.append(list(filter(lambda a: len(a), s)))

    # Get x and y data
    return [np.array(i) for i in shuffle_data(*split_data(data, seq_len))]

# Split data into train and test
x, y = load_data(VOCAB_THRESHOLD, SEQ_LEN)

# Data summary
print(f"x data shape: {x.shape}")
print(f"y data shape: {y.shape}")
print(f"Length of data: {len(x)}")

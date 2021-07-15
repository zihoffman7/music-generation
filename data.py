import os, re, random
import numpy as np

# Holds an element corresponding to a number
vocab = {}

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
            try:
                return list(filter(
                    # Remove invalid characters
                    lambda a: a != " " and a != "|" and a != "\n" and a != "/" and a != "" and a[0] != "%",
                    # Split abc at certain characters
                    re.split("(\s|\/|\||\n)",
                    # Merge the abc to one string
                    "".join(content[
                        # Cut the abc so only the first part remains
                        [i for i, s in enumerate(content) if "v:1\n" in s.lower() or "v:1 " in s.lower()][1] + 1:[i for i, s in enumerate(content) if "v:2\n" in s.lower() or "v:2 " in s.lower()][1]
                    ]).replace("\n", "").replace("\\", ""))
                ))
            except IndexError:
                return False

def split_data(data, seq_len):
    global vocab
    # Get number of elements in the data
    element_len = sum([len(i) for i in data])

    data_x = []
    data_y = []

    for d in data:
        # Extract a sequence of seq_len elements as x and the element after as y
        for i in range(0, element_len - seq_len):
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
    data_y = np.eye(len(data_y), len(vocab))[data_y]
    return data_x, data_y

def shuffle_data(x, y):
    # Zip data to prevent x and y data pairs from separating
    c = list(zip(x, y))
    random.shuffle(c)
    # Extract x and y pairs from shuffle
    return [np.array(i) for i in zip(*c)]

def load_data(vocab_threshold=5000, seq_len=50, **kwargs):
    global vocab
    # Will hold the processed train and test data
    data = []
    # Iterate through all abc files in song directory
    for file in os.listdir("data/abc"):
        # Extract the valid abc data contents
        s = parse_abc(f"data/abc/{os.fsdecode(file)}", **kwargs)
        if not s:
            continue
        for c, i in enumerate(s):
            # Ensure that the vocab threshold will not be exceeded
            if len(vocab) > vocab_threshold - 1:
                s.pop(c)
                break
            if len(i) and i not in vocab.keys():
                vocab[i] = len(vocab)
        data.append(s)
    print(f"Vocab size: {len(vocab)}")
    # Get x and y data
    return shuffle_data(*split_data(data, seq_len))

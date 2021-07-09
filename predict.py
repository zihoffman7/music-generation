from train_model import x, y, model, weights_save_path
from data import vocab
from convert import output_to_abc
import numpy as np

GEN_SIZE = 56
NUM_PREDICTIONS = 2

# Prevent horrible repetition
def sample(preds):
    exp_preds = np.exp(np.log(np.asarray(preds).astype("float64")))
    return np.argmax(np.random.multinomial(1, np.ndarray.flatten(exp_preds / np.sum(exp_preds)), 1))

model.load_weights(weights_save_path)
model.compile(loss="categorical_crossentropy", optimizer="adam")

reverse_vocab = {v: k for k, v in vocab.items()}

def predict(gen_size, key="C"):
    start = np.random.randint(0, len(x) - 1)
    pattern = x[start]
    output = []
    # generate characters
    for _ in range(gen_size):
        t = np.reshape(pattern, (1, len(pattern), 1))
        t = t / float(len(vocab))
        # Predict the next note
        prediction = model.predict(t, verbose=0)
        index = sample(prediction)
        result = reverse_vocab[index]
        output.append(result)
        # Pop the first element out and add in the generated element, like a queue
        pattern = pattern[1:]
        pattern = np.append(pattern, index / len(vocab))

    print(output)
    output_to_abc(output, key)

predict(GEN_SIZE)

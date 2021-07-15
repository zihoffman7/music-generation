from train_model import x, y, model, weights_save_path
from convert import output_to_abc
from data import vocab
import numpy as np

GEN_SIZE = 56

# Prevent horrible repetition
def sample(preds):
    exp_preds = np.exp(np.log(np.asarray(preds).astype("float64")))
    return np.argmax(np.random.multinomial(1, np.ndarray.flatten(exp_preds / np.sum(exp_preds)), 1))

# Load model from save
model.load_weights(weights_save_path)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Reverse the vocab for predictions
reverse_vocab = {v: k for k, v in vocab.items()}

def generate_melody(gen_size=56, key="C"):
    start = np.random.randint(0, len(x) - 1)
    pattern = x[start]
    output = []
    # generate characters
    for i in range(gen_size):
        # Shape data for the model
        t = np.reshape(pattern, (1, len(pattern), 1)) / len(vocab)
        # Predict the next note
        index = sample(model.predict(t, verbose=0))
        output.append(reverse_vocab[index])
        # Update the pattern queue
        pattern = pattern[1:]
        pattern = np.append(pattern, index / len(vocab))
    output_to_abc(output, key)
    elements_count = {}
    # iterating over the elements for frequency
    for element in output:
       # checking whether it is in the dict or not
       if element in elements_count:
          # incerementing the count by 1
          elements_count[element] += 1
       else:
          # setting the count to 1
          elements_count[element] = 1
    # printing the elements frequencies
    for key, value in elements_count.items():
       print(f"{key}: {value}")

generate_melody(GEN_SIZE)
# predict(GEN_SIZE)
# predict(GEN_SIZE)

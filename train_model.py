from keras.layers import Dropout, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from data import load_data
import numpy as np
import os
import matplotlib.pyplot as plt

# Length of patterns to be detected
SEQ_LEN = 10
# Max vocabulary that can be learned
VOCAB_THRESHOLD = 2000

BATCH_SIZE = 128
EPOCH_NUM = 700


# Load x and y data for the model
x, y = load_data(VOCAB_THRESHOLD, SEQ_LEN)

print(f"length of data: {len(x)}")
print((x.shape[1], x.shape[2]))
model = Sequential()
# First LSTM
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
# Second LSTM
model.add(LSTM(256))
model.add(Dropout(0.2))
# One output node for each vocab element
model.add(Dense(y.shape[1], activation="softmax"))
# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

weights_save_path = f"weights/{EPOCH_NUM}-epochs.hdf5"
# Save weights if not saved already
if not os.path.isfile(weights_save_path):
    checkpoint = ModelCheckpoint(weights_save_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
    callbacks_list = [checkpoint]

    history = model.fit(x, y, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend('train', loc='lower left')
    plt.savefig('acc.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend('train', loc='bottom left')
    plt.savefig('loss.png')

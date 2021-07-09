from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from data import load_data, vocab
import numpy as np
import os

# Length of patterns to be detected
SEQ_LEN = 10
# Max vocabulary that can be learned
VOCAB_THRESHOLD = 2000

BATCH_SIZE = 128
EPOCH_NUM = 50

# Split data into train and test
x, y = load_data(VOCAB_THRESHOLD, SEQ_LEN)

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

weights_save_path = "weights/50-epochs.hdf5"
# Save weights if not saved already
if not os.path.isfile(weights_save_path):
    checkpoint = ModelCheckpoint(weights_save_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
    callbacks_list = [checkpoint]

    model.fit(x, y, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, callbacks=callbacks_list)

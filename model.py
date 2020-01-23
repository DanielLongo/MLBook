from pickle import dump

import keras
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf

from data import load_document

tf.debugging.set_log_device_placement(True)

# for word, index in tokenizer.word_index.items():

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def main():
    epochs = 300
    seq_length = 50
    vocab_size = 16739
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    filename = "Gutenberg/txt/Ralph Waldo Emerson___Essays.txt"
    X, y = load_document(filename)
    # X = X.T
    y = indices_to_one_hot(y, vocab_size)
    print("X shape", X.shape)
    print("y shape", y.shape)
    model.fit(X, y, batch_size=128, epochs=300)
    # save the model to file
    model.save('model.h5')
    # dump(tokenizer, open('tokenizer.pkl', 'wb'))

def predict():
    model = load_model('model.h5')



if __name__ == "__main__":
    main()

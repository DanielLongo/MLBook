from pickle import dump

import keras
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf

from data import load_document, load_object, get_random_text

tf.debugging.set_log_device_placement(True)

# for word, index in tokenizer.word_index.items():
SEQEUNCE_LENGTH = 20
def detokenize(tokenizer, tokens):
    out = ""
    for token in tokens:
        for word, index in tokenizer.word_index.items():
            if token == index:
                out += " " + word
                break   
    return out

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def main():
    epochs = 500
    seq_length = SEQEUNCE_LENGTH
    # vocab_size = 10615
    vocab_size = 14170
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=seq_length))
    model.add(LSTM(400, return_sequences=True))
    model.add(LSTM(600, return_sequences=True))
    model.add(LSTM(400, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    # filename = "Gutenberg/txt/Ralph Waldo Emerson___Essays.txt"
    # filename = "Gutenberg/txt/Nathaniel Hawthorne___Biographical Stories.txt"
    filename = "Gutenberg/txt/Sinclair Lewis___Main Street.txt"
    X, y = load_document(filename)
    # X = X.T
    y = indices_to_one_hot(y, vocab_size)
    print("X shape", X.shape)
    print("y shape", y.shape)
    model.fit(X, y, batch_size=128, epochs=epochs)
    # save the model to file
    model.save('model.h5')

    # dump(tokenizer, open('tokenizer.pkl', 'wb'))

def genenerate_sequence(text, n_words, model_filename="model.h5", tokenizer_filename="tokenizer"):
    model = load_model(model_filename)
    tokenizer = load_object(tokenizer_filename)
    cur_text = text
    for i in range(n_words):
        # print("CUR TEXT", cur_text)
        cur_tokens = tokenizer.texts_to_sequences([cur_text])
        # print("CUR TOKENS", cur_tokens)
        # print("CUR TOKENS LENGHT", len(cur_tokens))
        cur_tokens = np.asarray(cur_tokens).reshape(1, SEQEUNCE_LENGTH)
        # print("cur tokens", cur_tokens.shape)
        cur_out_token = model.predict_classes(cur_tokens)
        cur_out_word = detokenize(tokenizer, cur_out_token)
        # print("out word", cur_out_word)
        text += cur_out_word
        cur_text = " ".join(text.split(" ")[-SEQEUNCE_LENGTH:])
        # print(len(text.split(" ")))

    print("STARTER TEXT: \n", " ".join(text.split(" ")[:SEQEUNCE_LENGTH]))
    print("GENERATED TEXT: \n", " ".join(text.split(" ")[SEQEUNCE_LENGTH:]))




if __name__ == "__main__":
    ### Train ###
    main()

    ### Genererate Sample Text ###
    # file = "Gutenberg/txt/Nathaniel Hawthorne___Biographical Stories.txt"
    file = "Gutenberg/txt/Sinclair Lewis___Main Street.txt"
    text = get_random_text(filename=file)
    print("RANDOM TEXT:", text)
    genenerate_sequence(text, 1000)

    ### Other ###
    # print(detokenize(load_object("tokenizer"), [47]))


import re
import numpy as np
from keras.preprocessing.text import Tokenizer
import _pickle as cPickle
import random

SEQUENCE_LENGTH = 20

def save_object(object, filename):
    with open(filename + ".pkl", "wb") as fid:
        cPickle.dump(object, fid)

def load_object(filename):
    with open(filename + ".pkl", "rb") as fid:
        object = cPickle.load(fid)
        return object

def create_text_chunks(text, chunk_length):
    # creates text chunks also with next word
    assert (type(text) == str), "Text input need to be a string "
    # print("type of text", type(text))
    split = text.split(" ")
    out_X = []
    out_y = []
    for i in range(0, len(split) - chunk_length, chunk_length):
        start = i
        end = i + chunk_length
        cur_chunk = split[start: end]
        out_X.append(cur_chunk)
        out_y.append(split[end])

    assert (len(out_X) == len(out_y)), "len of X and y should correspond"
    print('Len of chunks', len(out_X))
    return out_X, out_y


def get_tokenizer(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    save_object(tokenizer, "tokenizer")
    return tokenizer


def load_document(filename, tokenize=True):
    with open(filename, "r") as f:
        text = f.read()
    # print('text', text)
    cleaned_text = clean_text(text)
    print(type(cleaned_text))
    print("num unique words", )
    out_X, out_y = create_text_chunks(cleaned_text, SEQUENCE_LENGTH)
    
    if not tokenize:
        return out_X, out_y

    tokenizer = get_tokenizer(cleaned_text.split(" "))
    out_X = tokenizer.texts_to_sequences(out_X)
    out_y = tokenizer.texts_to_sequences(out_y)

    print("num words", len(tokenizer.word_index) + 1)
    print(len(tokenizer.word_index) + 1)

    # out_X = [tokenizer.texts_to_sequences(chunk) for chunk in out_X]
    # out_y = [tokenizer.texts_to_sequences(chunk) for chunk in out_y]
    print("sample X", out_X[0])
    print("sample y", out_y[0])
    out_X = np.asarray(out_X)
    out_y = np.asarray(out_y)
    return out_X, out_y


def clean_text(text):
    text = text.lower()
    text = re.sub(' +', ' ', text)
    text = text.replace(".", " PERIOD ")
    desired_chars = "abcdefghijklmnopqrstuvwxyzPERIOD "
    cleaned = ""
    for char in text:
        if char in desired_chars:
            cleaned += char
        else:
            cleaned += " "
            # print("char not desired", char)
    cleaned = re.sub(' +', ' ', cleaned)
    cleaned = cleaned.strip("")
    cleaned = cleaned.split(" ")
    out = []
    undesired_words = [""]
    for word in cleaned:
        print("word", word)
        if word in undesired_words:
            continue
        out.append(word)
    cleaned = " ".join(out)
    return cleaned

def get_random_text(tokenize=False, filename="Gutenberg/txt/Ralph Waldo Emerson___Essays.txt"):
    out_X,_ = load_document(filename, tokenize=tokenize)
    random_X = random.choice(out_X)
    if not tokenize:
        random_X = " ".join(random_X)
    return random_X


if __name__ == "__main__":
    # filename = "Gutenberg/txt/Ralph Waldo Emerson___Essays.txt"
    filename = "Gutenberg/txt/Sinclair Lewis___Main Street.txt"
    # filename = "Gutenberg/txt/Nathaniel Hawthorne___Biographical Stories.txt"
    out_X, out_y = load_document(filename)
    print("out_X", out_X.shape)
    print("out_y", out_y.shape)
    print("Finished data.py")

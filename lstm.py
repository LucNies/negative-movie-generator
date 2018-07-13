import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM as LSTM
from keras.callbacks import ModelCheckpoint
from keras import utils
from IPython import embed
import sys
from tqdm import tqdm
import os
seq_length = 100


def make_model(input_shape, output_dim, batch_size=128):
    print("model dim: ", input_shape, output_dim)
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, batch_size=batch_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def prepare_data():
    seq_length = 64
    # load ascii text and covert to lowercase
    filename = "wonderland.txt"
    raw_text = open(filename, encoding="utf8").read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = {number: character for (number, character) in zip(char_to_int.values(), char_to_int.keys())}

    n_chars = len(raw_text)
    n_vocab = len(chars)

    print("Total Characters: ", n_chars)
    # Convert text to integers and do one-hot encoding
    int_text = [char_to_int[c] for c in raw_text]
    int_text = utils.to_categorical(int_text)
    dataX = []
    dataY = []

    # Split sequences
    for i in range(0, n_chars - seq_length, 1):
        seq_in = int_text[i:i + seq_length]
        seq_out = int_text[i + seq_length]

        dataX.append(seq_in)
        dataY.append(seq_out)

    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    return raw_text, chars, dataX, dataY, int_to_char


def train(batch_size = 512):

    raw_text, chars, dataX, dataY, int_to_char = prepare_data()

    #Make model
    shape = (dataX.shape[1], dataX.shape[2])
    model = make_model(shape, dataY.shape[1], batch_size=None)

    filepath = os.path.join('models', "weights-improvement-{epoch:02d}-bigger.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(dataX, dataY, epochs=50, batch_size=512, callbacks=callbacks_list)


def generate(nth_iter):

    # load the network weights
    raw_text, chars, dataX, dataY, int_to_char = prepare_data()
    shape = (dataX.shape[1], dataX.shape[2])
    model = make_model(shape, dataY.shape[1])
    filename = os.path.join('models', "weights-improvement-{}-bigger.hdf5".format(nth_iter))
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = list(dataX[start].flatten())
    #embed()
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    output_string = ""
    # generate characters
    for i in tqdm(range(1000)):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(chars))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        output_string+=result
        #seq_in = [int_to_char[value] for value in pattern]
        #sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print(output_string)
    print("\nDone.")


if __name__ == "__main__":
    train()
    generate()

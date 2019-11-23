from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten

import constants as const


def get_trained_model(trainX, trainy):

    verbose, epochs, batch_size = 2, 20, 128
    n_input, n_output = trainX.shape[2], trainy.shape[1]

    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, int(const.BLOCK_SIZE / 4)
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_input))

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                              input_shape=(None, n_length, n_input)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(80))
    model.add(Dropout(0.3))
    model.add(Dense(60, activation='relu', name='feature_layer'))
    model.add(Dense(n_output, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model, history

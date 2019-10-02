from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten

import constants as const


def train_model(trainX, trainy):

    # define model
    verbose, epochs, batch_size = 2, 20, 128
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, int(const.NUM_FEATURES / 4)
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                              input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(80))
    model.add(Dropout(0.3))
    model.add(Dense(60, activation='relu', name='features_layer'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model, history

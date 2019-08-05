from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras import optimizers


def train_model(trainX, trainy):

    verbose, epochs, batch_size = 2, 16, 32
    n_timestamps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    input_shape = Input(shape=(n_timestamps, n_features))

    tower_11 = Conv1D(filters=40, kernel_size=6, strides=2, activation='relu')(input_shape)
    tower_12 = Conv1D(filters=60, kernel_size=3, strides=1, activation='relu')(tower_11)
    tower_1 = GlobalMaxPooling1D()(tower_12)

    tower_21 = Conv1D(filters=40, kernel_size=4, strides=2, activation='relu')(input_shape)
    tower_22 = Conv1D(filters=60, kernel_size=2, strides=1, activation='relu')(tower_21)
    tower_2 = GlobalMaxPooling1D()(tower_22)

    merged = concatenate([tower_1, tower_2])
    dropout = Dropout(0.15)(merged)
    out = Dense(60, activation='relu', name='features_layer')(dropout)
    out = Dense(n_outputs, activation='sigmoid')(out)

    model = Model(input_shape, out)
    optim = optimizers.Adam(lr=0.002, decay=1e-4)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model, history
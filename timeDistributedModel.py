import constants as const
import baseModel as base_model

from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten


class TimeDistributedModel(base_model.BaseModel):


    def __init__(self, model_name, input_shape, nb_classes, is_trainable = True):
        super().__init__(model_name, input_shape, nb_classes, is_trainable)


    def build_model(self, input_shape, nb_classes):
        self.verbose, self.epochs, self.batch_size = 2, 20, 128
        self.n_steps, self.n_length = 4, int(input_shape[1] / 4)

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                                input_shape=(None, self.n_length, input_shape[2]), trainable=self.is_trainable))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), trainable=self.is_trainable))
        model.add(TimeDistributed(Dropout(0.2), trainable=self.is_trainable))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2), trainable=self.is_trainable))
        model.add(TimeDistributed(Flatten(), trainable=self.is_trainable))
        model.add(LSTM(80, trainable=self.is_trainable))
        model.add(Dropout(0.3, trainable=self.is_trainable))
        model.add(Dense(60, activation='relu', name='feature_layer', trainable=self.is_trainable))
        model.add(Dense(nb_classes, activation='sigmoid', name='output_layer'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if const.VERBOSE == True:
            model.summary()

        return model


    def train_model(self, trainX, trainy):
        super().train_model()

        nb_epochs, mini_batch_size = 2000, 32
        # Fit network
        history = self.model.fit(trainX, trainy, batch_size=mini_batch_size, epochs=nb_epochs, validation_split=0.3, shuffle=True,
            verbose=const.VERBOSE, callbacks=self.callbacks)
        self.is_trained = True

        return history


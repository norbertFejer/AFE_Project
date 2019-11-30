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


    def __init__(self, model_name):
        super().__init__(model_name)

        self.verbose, self.epochs, self.batch_size = 2, 20, 128
        self.n_input = 2
        self.n_steps, self.n_length = 4, int(const.BLOCK_SIZE / 4)


    def create_model(self):

        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                                input_shape=(None, self.n_length, self.n_input)))
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        self.model.add(TimeDistributed(Dropout(0.2)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(80))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(60, activation='relu', name='feature_layer'))
        self.model.add(Dense(self.n_output, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train_model(self, trainX, trainy):
        super().train_model()

        trainX = trainX.reshape((trainX.shape[0], self.n_steps, self.n_length, self.n_input))

        # fit network
        history = self.model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return history


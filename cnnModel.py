import constants as const
import baseModel as base_model
import settings as stt

from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras import optimizers


class CNNmodel(base_model.BaseModel):


    def __init__(self, model_name):
        super().__init__(model_name)

        self.verbose, self.epochs, self.batch_size = 2, 16, 32


    def create_model(self, is_trainable = True):
        super().create_model(is_trainable)

        block_size, n_input = const.BLOCK_SIZE, 2

        input_shape = Input(shape=(block_size, n_input), name='input_layer')

        tower_11 = Conv1D(filters=40, kernel_size=6, strides=2, activation='relu', trainable=self.is_trainable)(input_shape)
        tower_12 = Conv1D(filters=60, kernel_size=3, strides=1, activation='relu', trainable=self.is_trainable)(tower_11)
        tower_1 = GlobalMaxPooling1D(trainable=self.is_trainable)(tower_12)

        tower_21 = Conv1D(filters=40, kernel_size=4, strides=2, activation='relu', trainable=self.is_trainable)(input_shape)
        tower_22 = Conv1D(filters=60, kernel_size=2, strides=1, activation='relu', trainable=self.is_trainable)(tower_21)
        tower_2 = GlobalMaxPooling1D(trainable=self.is_trainable)(tower_22)

        merged = concatenate([tower_1, tower_2], trainable=self.is_trainable)
        dropout = Dropout(0.15, trainable=self.is_trainable)(merged)
        out = Dense(60, activation='relu', name='feature_layer', trainable=self.is_trainable)(dropout)
        out = Dense(self.n_output, activation='sigmoid', name='output_layer')(out)

        self.model = Model(input_shape, out)
        optim = optimizers.Adam(lr=0.002, decay=1e-4)

        # print(self.model.summary())
        self.model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

        for l in self.model.layers:
            print(l.name, l.trainable)


    def train_model(self, trainX, trainy):
        super().train_model()

        # fit network
        history = self.model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        self.is_trained = True

        return history
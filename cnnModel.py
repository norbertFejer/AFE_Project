import constants as const
import baseModel as base_model

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
        block_size, n_input, n_output = const.BLOCK_SIZE, 2, 2

        input_shape = Input(shape=(block_size, n_input))

        tower_11 = Conv1D(filters=40, kernel_size=6, strides=2, activation='relu')(input_shape)
        tower_12 = Conv1D(filters=60, kernel_size=3, strides=1, activation='relu')(tower_11)
        tower_1 = GlobalMaxPooling1D()(tower_12)

        tower_21 = Conv1D(filters=40, kernel_size=4, strides=2, activation='relu')(input_shape)
        tower_22 = Conv1D(filters=60, kernel_size=2, strides=1, activation='relu')(tower_21)
        tower_2 = GlobalMaxPooling1D()(tower_22)

        merged = concatenate([tower_1, tower_2])
        dropout = Dropout(0.15)(merged)
        out = Dense(60, activation='relu', name='feature_layer')(dropout)
        out = Dense(n_output, activation='sigmoid')(out)

        self.model = Model(input_shape, out)
        optim = optimizers.Adam(lr=0.002, decay=1e-4)

        # print(self.model.summary())
        self.model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])


    def train_model(self, trainX, trainy):

        super().train_model()

        # fit network
        history = self.model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        return history
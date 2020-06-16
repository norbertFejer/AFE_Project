import tensorflow.keras as keras

import config.constants as const
import config.settings as stt
import models.baseModel as base_model


class Classifier_TCNN(base_model.BaseModel):


    def __init__(self, model_name, input_shape, nb_classes, is_trainable = True):
        super().__init__(model_name, input_shape, nb_classes, is_trainable)


    def build_model(self, input_shape, nb_classes):

        padding = 'valid'
        input_layer = keras.layers.Input(shape=(input_shape[1], input_shape[2]))

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        return model


    def train_model(self, trainX, trainy):

        super().train_model()

        mini_batch_size = 16
        nb_epochs = 2000

        # Fit network
        history = self.model.fit(trainX, trainy, batch_size=mini_batch_size, epochs=nb_epochs, validation_split=0.15, shuffle=False,
                                    verbose=const.VERBOSE, callbacks=self.callbacks)
        self.is_trained = True

        return history
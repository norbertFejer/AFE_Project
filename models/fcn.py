import tensorflow.keras as keras

import config.constants as const
import config.settings as stt
import models.baseModel as base_model


class Classifier_FCN(base_model.BaseModel):


    def __init__(self, model_name, input_shape, nb_classes, is_trainable = True):
        super().__init__(model_name, input_shape, nb_classes, is_trainable)


    def build_model(self, input_shape, nb_classes):

        input_layer = keras.layers.Input(shape=(input_shape[1], input_shape[2]))

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        if nb_classes == 2:
            model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['binary_accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])

        return model


    def train_model(self, trainX, trainy):
        super().train_model()

        batch_size = 16
        nb_epochs = 2000
        mini_batch_size = int(min(trainX.shape[0]/10, batch_size))

        # Fit network
        history = self.model.fit(trainX, trainy, batch_size=mini_batch_size, epochs=nb_epochs, validation_split=0.15, shuffle=False,
                                    verbose=const.VERBOSE, callbacks=self.callbacks)
        self.is_trained = True

        return history
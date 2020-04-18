import tensorflow.keras as keras

import config.constants as const
import config.settings as stt
import models.baseModel as base_model


class CNNmodel(base_model.BaseModel):


    def __init__(self, model_name, input_shape, nb_classes, is_trainable = True):
        super().__init__(model_name, input_shape, nb_classes, is_trainable)


    def build_model(self, input_shape, nb_classes):

        input_layer = keras.layers.Input(shape=(input_shape[1], input_shape[2]), name='input_layer')

        tower_11 = keras.layers.Conv1D(filters=40, kernel_size=6, strides=2, activation='relu', trainable=self.is_trainable)(input_layer)
        tower_12 = keras.layers.Conv1D(filters=60, kernel_size=3, strides=1, activation='relu', trainable=self.is_trainable)(tower_11)
        tower_1 = keras.layers.GlobalMaxPooling1D(trainable=self.is_trainable)(tower_12)

        tower_21 = keras.layers.Conv1D(filters=40, kernel_size=4, strides=2, activation='relu', trainable=self.is_trainable)(input_layer)
        tower_22 = keras.layers.Conv1D(filters=60, kernel_size=2, strides=1, activation='relu', trainable=self.is_trainable)(tower_21)
        tower_2 = keras.layers.GlobalMaxPooling1D(trainable=self.is_trainable)(tower_22)

        merged = keras.layers.concatenate([tower_1, tower_2], trainable=self.is_trainable)
        dropout = keras.layers.Dropout(0.15, trainable=self.is_trainable)(merged)
        feature_layer = keras.layers.Dense(60, activation='relu', name='feature_layer', trainable=self.is_trainable)(dropout)

        if nb_classes == 2:
            out_layer = keras.layers.Dense(nb_classes, activation='sigmoid', name='output_layer')(feature_layer)
        else:
            out_layer = keras.layers.Dense(nb_classes, activation='softmax', name='output_layer')(feature_layer)

        model = keras.models.Model(input_layer, out_layer)
        adam_optimizer = keras.optimizers.Adam(lr=0.002, decay=1e-4)

        if nb_classes == 2:
            model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['binary_accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['categorical_accuracy'])

        if const.VERBOSE == True:
            model.summary()

        return model


    def train_model(self, trainX, trainy):
        super().train_model()

        nb_epochs, mini_batch_size = 160, 32
        
        # Fit network
        history = self.model.fit(trainX, trainy, batch_size=mini_batch_size, epochs=nb_epochs, shuffle=False,
                                verbose=const.VERBOSE, callbacks=self.callbacks)
        self.is_trained = True

        return history
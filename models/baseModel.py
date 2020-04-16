import os
from abc import abstractmethod

from tensorflow.keras.models import load_model
import tensorflow.keras as keras

import config.settings as stt
import config.constants as const


class BaseModel:


    def __init__(self, model_name, input_shape, nb_classes, is_trainable = True, build = True):
        self.model = None
        self.model_name = model_name
        self.is_trained_model = False
        self.is_trainable = is_trainable

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)


    @abstractmethod
    def build_model(self, input_shape, nb_classes):
        """ Creates the given model. It has to be overwritten in derived class.

            Parameters:
                is_trainable (bool) - sets the model weights trainable property

            Returns:
                None
        """ 


    def __set_weights_from_pretrained_model(self, model_path):
        """ Loads weights from a pretrained model and sets the new model weights with these.

            Parameters:
                model_path (str) - pretrained model path

            Returns:
                None
        """ 
        try:
            old_model = load_model(model_path)
        except:
            raise Exception(model_path + ' model does not exist!')
        
        # The last layer weights will not be set
        for i in range(len(old_model.layers) - 1):
            self.model.layers[i].set_weights(old_model.layers[i].get_weights())
        print('setting weights###########################################################################################')
        print(stt.use_trainable_weights_for_transfer_learning)

    def train_model(self):
        """ Trains the actual model

            Parameters:
                None

            Returns:
                None
        """ 
        if stt.sel_method == stt.Method.TRAIN and stt.use_pretrained_weights_for_training_model:
            self.__set_weights_from_pretrained_model(const.TRAINED_MODELS_PATH + '/' + self.model_name)

        if stt.sel_method == stt.Method.TRANSFER_LEARNING:
            self.__set_weights_from_pretrained_model(const.TRAINED_MODELS_PATH + '/' + const.USED_MODEL_FOR_TRANSFER_LEARNING)

        #es = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=50, verbose=1, min_delta=0.1)
        #es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, min_delta=0.1)
        #es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
        #es = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=80, verbose=1)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = const.TRAINED_MODELS_PATH + "/best_" + self.model_name
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True, verbose=1)

        self.callbacks = [reduce_lr, model_checkpoint]
    
    
    def get_trained_model(self):
        """ Gets the trained model or None other case

            Parameters:
                None

            Returns:
                model (kereas Model) - trained model object
        """
        if self.is_trained_model:
            return self.model
        return None


    def save_model(self):
        """ Saves the trained model

            Parameters:
                None

            Returns:
                None
        """
        # Save the model
        if not os.path.exists(const.TRAINED_MODELS_PATH):
            os.makedirs(const.TRAINED_MODELS_PATH)

        self.model.save(const.TRAINED_MODELS_PATH + '/' + self.model_name)


    @staticmethod
    def predict_model(model_name, x_data):
        """ Predicts model result for given dataset

            Parameters:
                np.ndarry - dataset for prediction

            Returns:
                np.ndarry - the predicted dataset
        """
        model_path = const.TRAINED_MODELS_PATH + '/' + model_name
        model = load_model(model_path)

        return model.predict(x_data)
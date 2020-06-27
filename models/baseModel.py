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
                input_shape (int) - input dataset shape
                nb_classes (int) - number of output classes

            Returns:
                model (Keras.Model) - built model
        """ 


    def __set_weights_from_pretrained_model(self, model_path):
        print('setting weights from.........................................:', model_path)
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
        
        # The last layer weights will be ignored
        for i in range(len(old_model.layers) - 1):
            self.model.layers[i].set_weights(old_model.layers[i].get_weights())


    def train_model(self):
        """ Trains the actual model

            Parameters:
                None

            Returns:
                None
        """ 

        # Retrain model using pretrained weights
        if stt.sel_method == stt.Method.TRAIN and stt.use_pretrained_weights_for_training_model:
            self.__set_weights_from_pretrained_model(const.TRAINED_MODELS_PATH + '/' + self.model_name)

        # Performing transfer learning
        if stt.sel_method == stt.Method.TRANSFER_LEARNING:
            self.__set_weights_from_pretrained_model(const.TRAINED_MODELS_PATH + '/' + const.USED_MODEL_FOR_TRANSFER_LEARNING)

        # Path and model name for saving model
        file_path = const.TRAINED_MODELS_PATH + "/best_" + self.model_name
        # Callback function for saving the best model based, monitoring the loss value during training
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True, verbose=1)

        # We use different parameters for UserRecognitionType values
        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:

            if stt.sel_model == stt.Model.CLASSIFIER_MCDCNN:
                # Callback funtion for reducing learning rate
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.0001)

            if stt.sel_model == stt.Model.CLASSIFIER_FCN or stt.sel_model == stt.Model.CLASSIFIER_RESNET or stt.sel_model == stt.Model.CLASSIFIER_TCNN:
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
            
            # Callback funtion for performing early stopping
            es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=45, restore_best_weights=False, verbose=1)

            # For visualizing train only
            from time import time
            import datetime
            from tensorflow.keras.callbacks import TensorBoard
            log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard = TensorBoard(log_dir=log_dir, profile_batch=0, update_freq='epoch')

            self.callbacks = [reduce_lr, model_checkpoint, es, tensorboard]

        if stt.sel_user_recognition_type == stt.UserRecognitionType.IDENTIFICATION:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
            es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=80, restore_best_weights=False, verbose=1)
            #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
            self.callbacks = [model_checkpoint, es, reduce_lr]
    
    
    def get_trained_model(self):
        """ Returns the trained model or None if the model is not trained

            Parameters:
                None

            Returns:
                model (Kereas.Model) - trained model object
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
        # Checking if path exists
        if not os.path.exists(const.TRAINED_MODELS_PATH):
            os.makedirs(const.TRAINED_MODELS_PATH)

        self.model.save(const.TRAINED_MODELS_PATH + '/' + self.model_name)


    @staticmethod
    def predict_model(model_name, x_data):
        """ Predicts model using given dataset

            Parameters:
                np.ndarry - input dataset

            Returns:
                np.ndarry - the predicted results
        """
        model_path = const.TRAINED_MODELS_PATH + '/' + model_name
        model = load_model(model_path)

        return model.predict(x_data)
import settings as stt
import constants as const

import os
from keras.models import load_model


class BaseModel:


    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.is_trained_model = False

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            self.n_output = 2
        
        if stt.sel_user_recognition_type == stt.UserRecognitionType.IDENTIFICATION:
            self.n_output = len( stt.get_users() )


    def create_model(self, is_trainable):
        """ Creates the given model. It has to be overwritten in derived class.

            Parameters:
                is_trainable (bool) - sets the model weights trainable property

            Returns:
                None
        """ 
        self.is_trainable = is_trainable


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
        print('belepett........................................................................')
        # The last layer weights will not be set
        for i in range(len(old_model.layers) - 1):
            self.model.layers[i].set_weights(old_model.layers[i].get_weights())


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
    def predict_with_pretrained_model(model_name, x_data):
        """ Predicts model result for given dataset

            Parameters:
                np.ndarry - dataset for prediction

            Returns:
                np.ndarry - the predicted dataset
        """
        model_path = const.TRAINED_MODELS_PATH + '/' + model_name
        model = load_model(model_path)

        # Reshapes data for TIME_DISTRIBUTED model input
        if stt.sel_model == stt.Model.TIME_DISTRIBUTED:
            n_steps, n_length = 4, int(const.BLOCK_SIZE / 4)
            x_data = x_data.reshape((x_data.shape[0], n_steps, n_length, 2))

        return model.predict(x_data)
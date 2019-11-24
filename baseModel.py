import settings as stt
import constants as const

import os
from keras.models import load_model


class BaseModel:


    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name


    def set_weights_from_pretrained_model(self, model_path):

        if not os.path.exists(model_path):
            return

        old_model = load_model(model_path)

        for i in range(len(old_model.layers) - 1):
            self.model.layers[i].set_weights(old_model.layers[i].get_weights())


    def train_model(self):

        if stt.enable_transfer_learning:
            self.set_weights_from_pretrained_model(const.TRAINED_MODELS_PATH + '/' + self.model_name)

    
    def get_trained_model(self):
        return self.model


    def save_model(self):

        # Save the model
        if not os.path.exists(const.TRAINED_MODELS_PATH):
            os.makedirs(const.TRAINED_MODELS_PATH)

        self.model.save(const.TRAINED_MODELS_PATH + '/' + self.model_name)
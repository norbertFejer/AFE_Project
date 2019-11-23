import dataset as dset
import constants as const
import settings as stt

import timeDistributedModel
import cnnModel

import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
import os


class TrainModel:

    def __init__(self):
        self.dataset = dset.Dataset()


    if const.VERBOSE:
        def print_msg(self, msg):
            """ Prints the given message

                Parameters: msg (str)

                Returns:
                    void
            """
            print(msg)
    else:
        print_msg = lambda msg: None


    def __train_model_for_user(self, user, model_name):

        # Load input dataset
        self.print_msg('Loading train dataset...')
        
        trainX, trainy = self.dataset.create_train_dataset()
        trainy = to_categorical(trainy)

        self.print_msg('Loading train dataset finished.')
        self.print_msg(trainX.shape)

        model, history = self.__get_trained_model(trainX, trainy)

        # Save the model
        if not os.path.exists(const.TRAINED_MODELS_PATH):
            os.makedirs(const.TRAINED_MODELS_PATH)

        model.save(const.TRAINED_MODELS_PATH + '/' + model_name)

        return history


    def __get_model_0(self, trainX, trainy):
        return cnnModel.get_trained_model(trainX, trainy)


    def __get_model_1(self, trainX, trainy):
        return timeDistributedModel.get_trained_model(trainX, trainy)


    def __get_trained_model(self, trainX, trainy):
        
        switcher = {
            0: self.__get_model_0,
            1: self.__get_model_1
        }

        func = switcher.get(stt.sel_model.value, lambda: 'Not a valid model name!')
        return func(trainX, trainy)


    def __train_model_single_user(self, model_name):
        self.print_msg('Training model for user: ' + const.USER_NAME)
        self.__train_model_for_user(const.USER_NAME, model_name)


    def __train_model_all_user(self, model_name):
        usersArr = stt.get_users()

        for user in usersArr:
            self.print_msg('\nTraining model for user: ' + user + '\n')
            self.__train_model_for_user(user, model_name)
            self.print_msg('\nTraining model finished for user: ' + user + '\n')


    def train_model(self):

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            model_name = str(stt.sel_model) + '_' + str(stt.sel_dataset) + '_' + const.USER_NAME + '_' + str(const.BLOCK_SIZE) + '_' + str(const.BLOCK_NUM) + '_trained.h5'
        else:
            model_name = str(stt.sel_model) + ' ' + str(stt.sel_dataset) + '_identification_' + str(const.BLOCK_SIZE) + '_' + str(const.BLOCK_NUM) + '_trained.h5'

        if stt.sel_train_user_number == stt.TrainUserNumber.SINGLE:
            self.__train_model_single_user(model_name)
        else:
            self.__train_model_all_user(model_name)


    def __plot_train(self, history):

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['loss'])
        plt.title('Model training')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['Acc', 'Loss'], loc='upper left')
        plt.show()

import dataset as dset
import constants as const
import settings as stt

import timeDistributedModel
import cnnModel

import matplotlib.pyplot as plt
from keras.utils import to_categorical
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


    def __train_model_by_method(self, user, model_name):

        # Load input dataset
        self.print_msg('Loading train dataset...\n')
        
        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            trainX, trainy = self.dataset.create_train_dataset(user)
        else:
            trainX, trainy = self.dataset.create_train_dataset_for_identification()
        trainy = to_categorical(trainy)
        
        self.print_msg('Train dataset shape:')
        self.print_msg(trainX.shape)
        self.print_msg(trainy.shape)
        self.print_msg('\nLoading train dataset finished.')

        self.__train_selected_model(trainX, trainy, model_name)


    def __fit_and_save_model(self, model, trainX, trainy):
        model.train_model(trainX, trainy)
        model.save_model()


    def __get_model_0(self, trainX, trainy, model_name):
        cnn_model = cnnModel.CNNmodel(model_name)
        self.__fit_and_save_model(cnn_model, trainX, trainy)


    def __get_model_1(self, trainX, trainy, model_name):
        time_distributed_model = timeDistributedModel.TimeDistributedModel(model_name)
        self.__fit_and_save_model(time_distributed_model, trainX, trainy)


    def __train_selected_model(self, trainX, trainy, model_name):
        
        switcher = {
            0: self.__get_model_0,
            1: self.__get_model_1
        }

        train_model = switcher.get(stt.sel_model.value, lambda: 'Not a valid model name!')
        train_model(trainX, trainy, model_name)


    def __train_model_single_user(self, model_name):
        self.print_msg('\nTraining model for user: ' + const.USER_NAME + '...\n')
        model_name = const.USER_NAME + '_' + model_name
        self.__train_model_by_method(const.USER_NAME, model_name)
        self.print_msg('\nTraining model for user: ' + const.USER_NAME + ' finished.\n')


    def __train_model_all_user(self, model_name):
        usersArr = stt.get_users()

        for user in usersArr:
            self.print_msg('\nTraining model for user: ' + user + '\n')
            tmp_model_name = user + '_' + model_name
            self.__train_model_by_method(user, tmp_model_name)
            self.print_msg('\nTraining model finished for user: ' + user + '\n')


    def __train_model_identification(self, model_name):
        self.print_msg('Training model for identification...')
        model_name = 'identification_' + model_name
        self.__train_model_by_method(None, model_name)
        self.print_msg('Training model for identification finished...')


    def train_model(self):
        model_name = str(stt.sel_model) + '_' + str(stt.sel_dataset) + '_' + str(const.BLOCK_SIZE) + '_' + str(const.BLOCK_NUM) + '_trained.h5'

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:

            if stt.sel_train_user_number == stt.TrainUserNumber.SINGLE:
                self.__train_model_single_user(model_name)
            else:
                self.__train_model_all_user(model_name)      

        else:
            self.__train_model_identification(model_name)



    def __plot_train(self, history):

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['loss'])
        plt.title('Model training')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['Acc', 'Loss'], loc='upper left')
        plt.show()

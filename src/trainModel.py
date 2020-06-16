import os
import numpy as np

import matplotlib.pyplot as plt
from keras.utils import to_categorical

import src.dataset as dset
import config.constants as const
import config.settings as stt

import models.timeDistributedModel as timeDistributedModel
import models.cnnModel as cnnModel
import models.fcn as fcn
import models.resnet as resnet
import models.tcnn as tcnn

class TrainModel:

    def __init__(self):
        self.dataset = dset.Dataset.getInstance()


    if const.VERBOSE:
        def print_msg(self, msg):
            """ Prints the given message if VERBOSE is True

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
            trainX, trainy = self.dataset.create_train_dataset_for_authentication(user)
        else:
            trainX, trainy = self.dataset.create_train_dataset_for_identification()
        trainy = to_categorical(trainy)
        
        self.print_msg('Train dataset shape:')
        self.print_msg(trainX.shape)
        self.print_msg(trainy.shape)
        self.print_msg('\nLoading train dataset finished.')

        self.__train_selected_model(trainX, trainy, model_name)


    def __fit_and_save_model(self, model, trainX, trainy):

        history = model.train_model(trainX, trainy)
        model.save_model()

        # plt.plot(history.history['categorical_accuracy'])
        # plt.plot(history.history['val_categorical_accuracy'])
        
        # plt.title("MCD-CNN modell pontossága DFL adathalmaz felhasználóinak tanítása során", fontsize=30)
        # plt.xlabel("Korszak", fontsize=30)
        # plt.ylabel("Kategórikus pontosság (categorical accuracy)", fontsize=28)
        # plt.xticks(fontsize=28)
        # plt.yscale('log')
        # plt.yticks(fontsize=28)
        # plt.legend(['Tanítás', 'Validálás'], loc='lower right', fontsize=28)
        # plt.show()


    def __get_model_0(self, trainX, trainy, model_name, input_shape, nb_classes):
        """ Trains the CNN model

            Parameters:
                np.ndarray - input dataset
                np.ndarray - true labels
                str - model name

            Returns:
                None
        """
        cnn_model = cnnModel.CNNmodel(model_name, input_shape, nb_classes, stt.use_trainable_weights_for_transfer_learning)
        self.__fit_and_save_model(cnn_model, trainX, trainy)


    def __get_model_1(self, trainX, trainy, model_name, input_shape, nb_classes):
        """ Trains TIME_DISTRIBUTED model

            Parameters:
                np.ndarray - input dataset
                np.ndarray - true labels
                str - model name

            Returns:
                None
        """
        time_distributed_model = timeDistributedModel.TimeDistributedModel(model_name, input_shape, nb_classes, stt.use_trainable_weights_for_transfer_learning)
        self.__fit_and_save_model(time_distributed_model, trainX, trainy)


    def __get_model_2(self, trainX, trainy, model_name, input_shape, nb_classes):
        """ Trains Classifier_FCN model

            Parameters:
                np.ndarray - input dataset
                np.ndarray - true labels
                str - model name

            Returns:
                None
        """
        classifier_fcn = fcn.Classifier_FCN(model_name, input_shape, nb_classes, stt.use_trainable_weights_for_transfer_learning)
        self.__fit_and_save_model(classifier_fcn, trainX, trainy)


    def __get_model_3(self, trainX, trainy, model_name, input_shape, nb_classes):
        """ Trains Classifier_FCN model

            Parameters:
                np.ndarray - input dataset
                np.ndarray - true labels
                str - model name

            Returns:
                None
        """
        classifier_resnet = resnet.Classifier_RESNET(model_name, input_shape, nb_classes, stt.use_trainable_weights_for_transfer_learning)
        self.__fit_and_save_model(classifier_resnet, trainX, trainy)


    def __get_model_4(self, trainX, trainy, model_name, input_shape, nb_classes):
        """ Trains Classifier_TCNN model

            Parameters:
                np.ndarray - input dataset
                np.ndarray - true labels
                str - model name

            Returns:
                None
        """
        classifier_tcnn = tcnn.Classifier_TCNN(model_name, input_shape, nb_classes, stt.use_trainable_weights_for_transfer_learning)
        self.__fit_and_save_model(classifier_tcnn, trainX, trainy)


    def __train_selected_model(self, trainX, trainy, model_name):
        """ Getting model by name and training it

            Parameters:
                np.ndarray - input dataset
                np.ndarray - true labels
                str - model name

            Returns:
                None
        """
        
        switcher = {
            0: self.__get_model_0,
            1: self.__get_model_1,
            2: self.__get_model_2,
            3: self.__get_model_3,
            4: self.__get_model_4
        }

        train_model = switcher.get(stt.sel_model.value, lambda: 'Not a valid model name!')

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            nb_classes = 2

        if stt.sel_user_recognition_type == stt.UserRecognitionType.IDENTIFICATION:
            nb_classes = len( stt.get_users() )

        train_model(trainX, trainy, model_name, trainX.shape, nb_classes)


    def __train_model_single_user(self, model_name):
        self.print_msg('\nTraining model for user: ' + const.USER_NAME + '...\n')
        tmp_model_name = const.USER_NAME + '_' + model_name
        self.__train_model_by_method(const.USER_NAME, tmp_model_name)
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
        """ Trains the model with the selected params

            Parameters:
                None

            Returns:
                None
        """
        model_name = str(stt.sel_model) + '_' + str(stt.sel_dataset) + '_' + str(const.BLOCK_SIZE) + '_' + str(stt.BLOCK_NUM) + '_trained.hdf5'

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

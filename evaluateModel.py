import constants as const
import settings as stt
import dataset as dset

from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from keras.models import Model
import numpy as np


class EvaluateModel:


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


    # Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from predicted scores
    def __get_auc_result(self, model, testX, y_true):

        y_scores = model.predict(testX)
        return roc_auc_score(y_true, y_scores)


    # Computes Accuracy
    def __get_acc_result(self, model, testX, y_true):

        y_pred = np.argmax( model.predict(testX), axis=1)
        y_true = np.argmax( y_true, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        
        return accuracy


    def __get_confusion_matrix(self, model, testX, y_true):

        y_pred = np.argmax( model.predict(testX), axis=1)
        y_true = np.argmax( y_true, axis=1)
        conf_matrix = confusion_matrix(y_true, y_pred)

        return conf_matrix


    def __get_evaluation_score(self, arg, model, testX, y_true):
        switcher = { 
            1: self.__get_acc_result,
            2: self.__get_auc_result,
            3: self.__get_confusion_matrix 
        } 
    
        func = switcher.get(arg, lambda: "Wrong evaluation metric!")
        return func(model, testX, y_true)


    def __evaluate_model_by_metrics(self, model, testX, y_true):

        for metric in stt.sel_evaluation_metrics:
            self.print_msg('\n' + str(metric) + ' value:')
            print( self.__get_evaluation_score(metric.value, model, testX, y_true) )


    def __evaluate_model_by_user(self, user, model_name):
        self.print_msg('\nEvaluatig model for user: ' + user + '...')

        model_path = const.TRAINED_MODELS_PATH + '/' + user + '_' + model_name
        model = load_model(model_path)

        testX, y_true = self.dataset.create_test_dataset(user)
        self.print_msg('\nTest dataset shape: ')
        self.print_msg(testX.shape)
        self.print_msg(y_true.shape)

        y_true = to_categorical(y_true)
        
        self.__evaluate_model_by_metrics(model, testX, y_true)
        self.print_msg('\nEvaluatig model for user ' + user + ' finished.\n')


    def __evaluate_model_action_based_singel_user(self, model_name):
        self.__evaluate_model_by_user(const.USER_NAME, model_name)


    def __evaluate_model_action_based_all_user(self, model_name):
        userArr = stt.get_users()

        for user in userArr:
            self.__evaluate_model_by_user(user, model_name)
            
            
    def __action_based_evaluation(self, model_name):

        if stt.sel_evaluate_user_number == stt.EvaluateUserNumber.SINGLE:
            self.__evaluate_model_action_based_singel_user(model_name)
        else:
            self.__evaluate_model_action_based_all_user(model_name)


    def evaluate_model(self):

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            model_name = str(stt.sel_model) + '_' + str(stt.sel_dataset) + '_' + str(const.BLOCK_SIZE) + '_' + str(const.BLOCK_NUM) + '_trained.h5'

            if stt.sel_evaluation_type == stt.EvaluationType.ACTION_BASED:
                self.__action_based_evaluation(model_name)
            else:
                print('session based')

        else:
            model_name = 'identification_' + str(stt.sel_model) + '_' + str(stt.sel_dataset) + '_' + str(const.BLOCK_SIZE) + '_' + str(const.BLOCK_NUM) + '_trained.h5'
            print('identification')
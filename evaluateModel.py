import constants as const
import settings as stt
import dataset as dset
import baseModel as base_model

from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import os


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
    def __get_auc_result(self, model_name, testX, y_true):

        y_scores = base_model.BaseModel.predict_with_pretrained_model(model_name, testX)
        return roc_auc_score(y_true, y_scores)


    # Computes Accuracy
    def __get_acc_result(self, model_name, testX, y_true):

        y_pred = np.argmax( base_model.BaseModel.predict_with_pretrained_model(model_name, testX), axis=1)
        y_true = np.argmax( y_true, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        
        return accuracy


    def __get_confusion_matrix(self, model_name, testX, y_true):

        y_pred = np.argmax( base_model.BaseModel.predict_with_pretrained_model(model_name, testX), axis=1)
        y_true = np.argmax( y_true, axis=1)
        conf_matrix = confusion_matrix(y_true, y_pred)

        return conf_matrix


    def __get_evaluation_score(self, arg, model_name, testX, y_true):
        switcher = { 
            1: self.__get_acc_result,
            2: self.__get_auc_result,
            3: self.__get_confusion_matrix 
        } 
    
        func = switcher.get(arg, lambda: "Wrong evaluation metric!")
        return func(model_name, testX, y_true)


    def __evaluate_model_by_metrics(self, model_name, testX, y_true):
        results = {}

        for metric in stt.sel_evaluation_metrics:
            self.print_msg('\n' + str(metric) + ' value:')
            value = self.__get_evaluation_score(metric.value, model_name, testX, y_true)
            results[str(metric)] = value
            self.print_msg(value)

        if stt.print_evaluation_results_to_file:
            self.__print_results_to_file(results, model_name)


    def __evaluate_model_by_method(self, user, model_name):

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            testX, y_true = self.dataset.create_test_dataset(user)
        else:
            testX, y_true = self.dataset.create_test_dataset_for_identification()
        y_true = to_categorical(y_true)

        self.print_msg('\nTest dataset shape: ')
        self.print_msg(testX.shape)
        self.print_msg(y_true.shape)
        
        self.__evaluate_model_by_metrics(model_name, testX, y_true)


    def __evaluate_model_action_based_single_user(self, model_name):
        self.print_msg('\nEvaluating model for user: ' + const.USER_NAME + '...')
        model_name = const.USER_NAME + '_' + model_name
        self.__evaluate_model_by_method(const.USER_NAME, model_name)
        self.print_msg('\nEvaluating model finished.\n')


    def __evaluate_model_action_based_all_user(self, model_name):
        userArr = stt.get_users()

        for user in userArr:
            self.print_msg('\nEvaluatig model for user: ' + user + '...')
            tmp_model_name = user + '_' + model_name
            self.__evaluate_model_by_method(user, tmp_model_name)
            self.print_msg('\nEvaluatig model finished.\n')
            
            
    def __action_based_evaluation(self, model_name):

        if stt.sel_evaluate_user_number == stt.EvaluateUserNumber.SINGLE:
            self.__evaluate_model_action_based_single_user(model_name)
        else:
            self.__evaluate_model_action_based_all_user(model_name)


    def __evaluate_model_for_identification(self, model_name):
        model_name = 'identification_' + model_name
        self.__evaluate_model_by_method(None, model_name)


    def evaluate_model(self):
        model_name = str(stt.sel_model) + '_' + str(stt.sel_dataset) + '_' + str(const.BLOCK_SIZE) + '_' + str(const.BLOCK_NUM) + '_trained.h5'

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:

            if stt.sel_evaluation_type == stt.EvaluationType.ACTION_BASED:
                self.__action_based_evaluation(model_name)
            else:
                print('session based')

        else:
            self.__evaluate_model_for_identification(model_name)


    def __print_results_to_file(self, results, file_name):

        if not os.path.exists(const.RESULTS_PATH):
            os.makedirs(const.RESULTS_PATH)

        file = open(const.RESULTS_PATH + '/' + file_name + '.csv', 'w')
        
        for key, value in results.items():
            file.write(str(key) + ':\n')
            file.write(str(value) + '\n')

        file.close()


if __name__ == "__main__":
    em = EvaluateModel()
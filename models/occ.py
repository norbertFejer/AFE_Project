import os
import csv

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

import config.settings as stt
import config.constants as const
import src.dataset as dset

# Setting random states to get reproducible results
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)


class Classifier(Enum):
    OCSVM = 'ocsvm'
    IFOREST = 'iforest'
    EllipticEnvelope = 'ellipticEnvelope'
    LocalOutlierFactor = 'localOutlierFactor'

# Global parameters
results = {}
classifier = None


# Defines the selected classifier for performing user identification
sel_classifier = Classifier.OCSVM


def fit_selected_classifier(X_data):
    """ Fits the selected classifier

        Parameters:
            X_data (np.ndarray) - input dataset

        Returns:
            None
    """
    global classifier

    if sel_classifier == Classifier.OCSVM:
        #classifier = OneClassSVM(kernel='sigmoid', gamma='scale', nu=0.1).fit(X_data)
        classifier = OneClassSVM(gamma='scale').fit(X_data)

    if sel_classifier == Classifier.IFOREST:
        classifier = IsolationForest(random_state=0, contamination=0.1).fit(X_data)

    if sel_classifier == Classifier.EllipticEnvelope:
        classifier = IsolationForest(random_state=0, contamination=0.1).fit(X_data)

    if sel_classifier == Classifier.LocalOutlierFactor:
        classifier = LocalOutlierFactor(contamination=0.1, novelty=True).fit(X_data)


def aggregate_blocks(y_pred):
    """ Aggregate blocks for evaluating model using multiple blocks

        Parameters:
            y_pred (np.ndarray) - predicted result for each block

        Returns:
            None
    """
    
    if const.AGGREGATE_BLOCK_NUM == 1:
        return y_pred

    if stt.sel_authentication_type == stt.AuthenticationType.ONE_CLASS_CLASSIFICATION:

        y_pred = y_pred.astype(float)
        # Aggregating positive class values
        for i in range(const.TRAIN_TEST_SPLIT_VALUE - const.AGGREGATE_BLOCK_NUM + 1):
            y_pred[i] = np.average(y_pred[i : i + const.AGGREGATE_BLOCK_NUM], axis=0)

        # Aggregating negative class values
        for i in range(const.TRAIN_TEST_SPLIT_VALUE, len(y_pred) - const.AGGREGATE_BLOCK_NUM + 1):
            y_pred[i] = np.average(y_pred[i : i + const.AGGREGATE_BLOCK_NUM], axis=0)

    return y_pred


def get_auc_result(testX, y_true):
    """ Return predicted AUC value for each block

        Parameters:
            testX (np.ndarray) - test dataset
            y_true (np.ndarray) - true labels

        Returns:
            float - AUC value
    """
    y_pred = classifier.score_samples(testX)
    y_pred = aggregate_blocks(y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=0)

    return metrics.auc(fpr, tpr)


def train_test_classifier(user):
    """ Train and test classifier

        Parameters:
            user (str) - username for training and evaluating model

        Returns:
            None
    """
    global results

    dataset = dset.Dataset.getInstance()

    stt.sel_method = stt.Method.TRAIN
    # Create train dataset
    X_data, _ = dataset.create_train_dataset_for_authentication(user)
    print('Train dataset shape: ', X_data.shape)

    print('Training model', sel_classifier.value, 'for user:', user, '...')
    # Fit selected network
    fit_selected_classifier(X_data)

    stt.sel_method = stt.Method.EVALUATE
    # Create test dataset
    X_data, y_true = dataset.create_test_dataset_for_authentication(user)
    print('Test dataset shape:', X_data.shape)

    print('Evaluate model', sel_classifier.value, '...')
    # Getting AUC from predicted values
    results[user] = get_auc_result(X_data, y_true)
    print('Evaluated AUC for user:', user, ' ', results[user], '\n')


def print_result_to_file(file_name):
    """ Save evaluation results to file

        Parameters:
            file_name (str) - filename

        Returns:
            None
    """

    if not os.path.exists(const.RESULTS_PATH):
            os.makedirs(const.RESULTS_PATH)

    file = open(const.RESULTS_PATH + '/' + file_name + '.csv', 'w')
    file.write('username,AUC\n')
    
    # Iterating through each user's AUC values
    for user, value in results.items():
        file.write(str(user) + ',' + str(value) + '\n')

    file.close()


def print_dataset_to_csv(data):
    """ Print OCC input dataset to csv file

        Parameters:
            data (np.ndarray) - input dataset

        Returns:
            None
    """

    with open('occ_dataset.csv', 'w', newline='') as f:
 
        for i in range(data.shape[0]):

            for j in range(data.shape[1]):
                f.write(str(data[i, j]))
                f.write(',')

            f.write('\n')


def read_dataset(filename):
    return pd.read_csv(filename)



def main():

    if stt.sel_train_user_number == stt.TrainUserNumber.ALL:
        for user in stt.get_users():
            train_test_classifier(user)

    if stt.sel_train_user_number == stt.TrainUserNumber.SINGLE:
        train_test_classifier(const.USER_NAME)

    print_result_to_file('authentication_' + sel_classifier.value + '_' + str(stt.sel_dataset) + '_' + stt.sel_occ_features.value)



if __name__ == "__main__":
    main()
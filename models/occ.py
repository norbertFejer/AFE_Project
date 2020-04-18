import os

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


results = {}
classifier = None

sel_classifier = Classifier.IFOREST


def fit_selected_classifier(X_data):
    global classifier

    if sel_classifier == Classifier.OCSVM:
        classifier = OneClassSVM(kernel='sigmoid', gamma='scale', nu=0.1).fit(X_data)

    if sel_classifier == Classifier.IFOREST:
        classifier = IsolationForest(random_state=0, contamination=0.1).fit(X_data)

    if sel_classifier == Classifier.EllipticEnvelope:
        classifier = IsolationForest(random_state=0, contamination=0.1).fit(X_data)

    if sel_classifier == Classifier.LocalOutlierFactor:
        classifier = LocalOutlierFactor(contamination=0.1, novelty=True).fit(X_data)


def aggregate_blocks(y_pred):

        if const.AGGREGATE_BLOCK_NUM == 1:
            return y_pred

        y_pred = y_pred.astype(float)
        for i in range(len(y_pred) - const.AGGREGATE_BLOCK_NUM + 1):
            y_pred[i] = np.average(y_pred[i : i + const.AGGREGATE_BLOCK_NUM])

        return y_pred


def train_test_classifier(user):
    global results

    dataset = dset.Dataset.getInstance()

    stt.sel_method = stt.Method.TRAIN
    X_data, _ = dataset.create_train_dataset_for_authentication(user)
    print('Train dataset shape: ', X_data.shape)

    print('Training model', sel_classifier.value, 'for user', user, '...')
    fit_selected_classifier(X_data)

    stt.sel_method = stt.Method.EVALUATE
    X_data, y_true = dataset.create_test_dataset_for_authentication(user)
    print('Test dataset shape:', X_data.shape)

    print('Evaluate model', sel_classifier.value, '...')
    y_pred = classifier.predict(X_data)

    y_pred = aggregate_blocks(y_pred)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    results[user] = metrics.auc(fpr, tpr)
    print('Evaluated AUC for user:', user, ' ', results[user], '\n')


def print_result_to_file(file_name):

    if not os.path.exists(const.RESULTS_PATH):
            os.makedirs(const.RESULTS_PATH)

    file = open(const.RESULTS_PATH + '/' + file_name + '.csv', 'w')
    file.write('username,AUC\n')
    
    for user, value in results.items():
        file.write(str(user) + ',' + str(value) + '\n')

    file.close()


def main():

    if stt.sel_train_user_number == stt.TrainUserNumber.ALL:
        for user in stt.get_users():
            train_test_classifier(user)

    if stt.sel_train_user_number == stt.TrainUserNumber.SINGLE:
        train_test_classifier(const.USER_NAME)

    print_result_to_file('authentication_' + sel_classifier.value + '_' + str(stt.sel_dataset) + '_' + stt.sel_occ_features.value)



if __name__ == "__main__":
    main()
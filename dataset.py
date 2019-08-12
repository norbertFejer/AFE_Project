import numpy as np
import pandas as pd
import glob
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import math

import constants as const
import settings


def load_file(filepath):
    return pd.read_csv(filepath).values


# load a single file as a numpy array
# contains only x, y and client timestamps
def load_relevant_data_from_file(filepath):

     if const.SAMPLES_NUM == 'ALL':
         return pd.read_csv(filepath, usecols=['x', 'y', 'client timestamp'])[['x', 'y', 'client timestamp']].values

     number_of_rows = const.SAMPLES_NUM * const.NUM_FEATURES
     return pd.read_csv(filepath, usecols=['x', 'y', 'client timestamp'],
                        nrows=number_of_rows)[['x', 'y', 'client timestamp']].values


# load a list of files and return as a 3d numpy array
def load_dataset_group(filename, filepath, numberOfRows):

    loaded = list()

    # iterating through every file in the given folder
    for file in glob.iglob(filepath + '/' + filename + '/*'):
        data = load_relevant_data_from_file(file)
        velocities = get_velocity_from_data(data)
        # normalizing data to [0, 1]
        normalized_velocities = normalize_data(velocities, 'builtin')

        # checking if normalized_velocities is empty
        if normalized_velocities.shape[1] != 0:
            loaded.extend(normalized_velocities)

            if len(loaded) > numberOfRows:
                return loaded[:numberOfRows]

    return loaded


# returns the final dataset with the given form (size)
# divides entire array to a given chunk of samples (defined by n_features)
def get_partitioned_dataset(dataset, n_features):

    # means that not enough samples is available
    if dataset.shape[0] < n_features:
        return np.empty([1, 0])

    values = list()

    # slicing array to n_feature size chunks
    for i in range(int(len(dataset) / n_features)):
        int_start = i * n_features
        int_stop = i * n_features + n_features
        values.append(dataset[int_start:int_stop, :].T)

    return np.transpose(np.asarray(values), (0, 2, 1))


# return velocities from data
def get_velocity_from_data(data):

    data = remove_idle_state(data)

    velocity = list()
    # period between two consecutive timestamp if dt is 0
    s_dt = 0.01

    for i in range(len(data) - 1):

        dt = data[i + 1][2] - data[i][2]

        tmp_x = 0
        tmp_y = 0
        if (dt != 0):
            tmp_x = (abs(data[i + 1][0] - data[i][0])) / dt
            tmp_y = (abs(data[i + 1][1] - data[i][1])) / dt
        else:
            tmp_x = (abs(data[i + 1][0] - data[i][0])) / s_dt
            tmp_y = (abs(data[i + 1][1] - data[i][1])) / s_dt

        velocity.append([tmp_x, tmp_y])

    return velocity


# removing states those have bigger difference than STATELESS_TIME
def remove_idle_state(data):

    i = 0
    tmp_dataset = list()

    while (i + const.NUM_FEATURES) <= data.shape[0]:

        hasStatelesTime = False
        for j in range(i, i + const.NUM_FEATURES - 1):

            # checking if given difference is bigger than STATELESS_TIME
            # it means that the mouse dynamic is negligible
            if data[j + 1][2] - data[j][2] > const.STATELESS_TIME or data[j][2] < 0:
                i = j + 1
                hasStatelesTime = True
                break

        if hasStatelesTime == False:
            tmp_dataset.extend(data[i:i + const.NUM_FEATURES, :])
            i = i + const.NUM_FEATURES

    return np.asarray(tmp_dataset)


# normalizing data with the given method
def normalize_data(dataset, method):

    if method == 'builtin':
        return normalize_data_builtin(dataset)
    else:
        return normalize_data_user_defined(dataset)


# this uses the default L1 and L2 normalization
def normalize_data_builtin(data):

    # L2 normalization
    return tf.keras.utils.normalize(data, axis=-1, order=2)

    # L1 normalization
    # return tf.keras.utils.normalize(data, axis=-1, order=1)


# searching the maximum and dividing with it
def normalize_data_user_defined(data):

    data = np.asarray(data)
    max_tuple = np.amax(data, axis=0)
    data[:, 0] /= max_tuple[0]
    data[:, 1] /= max_tuple[1]

    return data


# file - filename
# filepath - the given file path
# n_features - this is the feature number for one chunk
def load_positive_dataset(file, filepath, numberOfSamples):

    dataset = np.asarray(load_dataset_group(file, filepath, numberOfSamples))

    # converting array to n_features number of chunks
    return get_partitioned_dataset(dataset, const.NUM_FEATURES)


# creating output values:

# defines an array with the given size and value
def create_train_label(n_timestamps, value):
    return np.full((n_timestamps, 1), value)


def create_train_input(userName, filePath):

    if const.SAMPLES_NUM == 'ALL':
        number_of_samples = math.inf
    else:
        number_of_samples = const.SAMPLES_NUM * const.NUM_FEATURES

    if settings.balanceType == settings.Balance.POSITIVE:
         dset_positive, dset_negative = load_positive_balanced(userName, filePath, number_of_samples)

    if settings.balanceType == settings.Balance.NEGATIVE:
        dset_positive, dset_negative = load_negative_balanced(userName, filePath, number_of_samples)

    if settings.selectedDataSet == settings.Dataset.DFL:
        label_positive = create_train_label(dset_positive.shape[0], 0)  # 0 valid user
        dset_positive, X_test, label_positive, y_test = train_test_split(dset_positive, label_positive,
                                                                         test_size=const.TRAIN_SPLIT_VALUE,
                                                                         random_state=const.RANDOM_STATE)
        dset_negative = dset_negative[:dset_positive.shape[0]]
    else:
        label_positive = create_train_label(dset_positive.shape[0], 0)  # 0 valid user

    label_negative = create_train_label(dset_negative.shape[0], 1)  # 1 intrusion detected (is illegal)

    return np.concatenate((dset_positive, dset_negative), axis=0), \
           np.concatenate((label_positive, label_negative), axis=0)

def load_positive_balanced(userName, filePath, numberOfRows):

    # NUM_FEATURES is the feature num in a sample
    dset_positive = load_positive_dataset(userName, filePath, numberOfRows)

    if isinstance(const.SAMPLES_NUM, int) and dset_positive.shape[0] < const.SAMPLES_NUM:

        while dset_positive.shape[0] < const.SAMPLES_NUM:
            dset_positive = np.concatenate((dset_positive, dset_positive), axis=0)

        dset_positive = dset_positive[:const.SAMPLES_NUM]

    numberOfUsers = len(os.listdir(filePath))
    numberOfSamples = int(dset_positive.shape[0] * const.NUM_FEATURES / (numberOfUsers - 1))

    dset_negative = load_negative_dataset(userName, filePath, numberOfSamples)

    if dset_negative.shape[0] < dset_positive.shape[0]:

        while dset_negative.shape[0] < dset_positive.shape[0]:
            dset_negative = np.concatenate((dset_negative, dset_negative), axis=0)

        dset_negative = dset_negative[:dset_positive.shape[0]]

    return dset_positive, dset_negative


def load_negative_balanced(userName, filePath, numberOfSamples):

    dset_negative = load_negative_dataset(userName, filePath, numberOfSamples)

    numberOfRows = (len(os.listdir(filePath)) - 1) * const.SAMPLES_NUM

    if isinstance(const.SAMPLES_NUM, int) and dset_negative.shape[0] < numberOfRows:

        while dset_negative.shape[0] < numberOfRows:
            dset_negative = np.concatenate((dset_negative, dset_negative), axis=0)

        dset_negative = dset_negative[:numberOfRows]

    dset_positive = load_positive_dataset(userName, filePath, dset_negative.shape[0])

    if dset_positive.shape[0] < dset_negative.shape[0]:

        while dset_positive.shape[0] < dset_negative.shape[0]:
            dset_positive = np.concatenate((dset_positive, dset_positive), axis=0)

        dset_positive = dset_positive[:dset_negative.shape[0]]

    return dset_positive, dset_negative


def load_negative_group_samples(filename, filepath, numSamples):

    loadedData = np.empty([1, 2])

    # getting all subfolders
    users = os.listdir(filepath)
    # removing from list the user's session what we use during training our model
    users.remove(filename)

    for user in users:
        tmpData = load_dataset_group(user, filepath, numSamples)

        if len(tmpData) != 0:
            loadedData = np.concatenate((loadedData, tmpData), axis=0)

    return loadedData


# load given file
def load_random_file_with_velocities(filepath):

    data = load_relevant_data_from_file(filepath)
    velocities = get_velocity_from_data(data)
    velocities = normalize_data(velocities, 'builtin')

    return velocities


def load_negative_dataset(file, filepath, numberOfSamples):

    dataset = np.asarray(load_negative_group_samples(file, filepath, numberOfSamples))

    return get_partitioned_dataset(dataset, const.NUM_FEATURES)


def get_action_based_test_data_with_labels(sessionName, testFilesPath, labelsPath, n_features):

    labels = load_file(labelsPath)

    loadedData = list()
    output_labels = list()

    finalPath = testFilesPath + sessionName + '/*'

    for file in glob.iglob(finalPath):

        # if given file is labeled
        # (not every test file in given user folder is labeled)
        if file[len(file) - 18: len(file)] in labels:
            tmpData = load_relevant_data_from_file(file)
            velocities = get_velocity_from_data(tmpData)
            velocities = normalize_data(velocities, 'builtin')

            label_len = int(np.asarray(velocities).shape[0] / n_features)
            m_len = label_len * n_features
            loadedData.extend(np.asarray(velocities)[:m_len, :])

            # given session value
            # 0 is valid user
            # 1 if intrusion detected
            val = np.where(labels == file[len(file) - 18: len(file)])
            output_labels.extend(create_train_label(label_len, int(labels[val[0]][0][1])))

    return np.asarray(loadedData), np.asarray(output_labels)


def load_action_based_test_dataset_balabit(sessionName, testFilesPath, labelsPath, n_features):

    dataset, labels = get_action_based_test_data_with_labels(sessionName, testFilesPath, labelsPath, n_features)

    return get_partitioned_dataset(dataset, n_features), labels


def create_test_dataset(user):

    dset_tmp, labels = [], []

    if settings.selectedDataSet == settings.Dataset.BALABIT:
        dset_tmp, labels = load_action_based_test_dataset_balabit(user, const.TEST_FILES_PATH, const.TEST_LABELS_PATH, const.NUM_FEATURES)

    if settings.selectedDataSet == settings.Dataset.DFL:
        dset_tmp, labels = load_action_based_test_dataset_dfl(user, const.TRAINING_FILES_PATH)

    return get_shaped_dataset(dset_tmp, labels)


def load_action_based_test_dataset_dfl(sessionName, filePath):

    # NUM_FEATURES is the feature num in a sample
    dset_positive = load_positive_dataset(sessionName, filePath, const.NUM_FEATURES * const.SAMPLES_NUM)
    label_positive = create_train_label(dset_positive.shape[0], 0)  # 0 valid user

    X_aux, dset_positive, y_aux, label_positive = train_test_split(dset_positive, label_positive,
                                                                       test_size=const.TRAIN_SPLIT_VALUE,
                                                                       random_state=const.RANDOM_STATE)

    dset_negative = load_negative_dataset(sessionName, filePath, dset_positive.shape[0] * const.NUM_FEATURES)
    label_negative = create_train_label(dset_negative.shape[0], 1)  # 1 intrusion detected (is illegal)

    return np.concatenate((dset_positive, dset_negative), axis=0), \
           np.concatenate((label_positive, label_negative), axis=0)


# it shapes dataset to given model input dimensions
def get_shaped_dataset(dataset, labels):

    if settings.selectedModel == settings.Model.CNN:
        return dataset, labels

    if settings.selectedModel == settings.Model.TIME_DISTRIBUTED:
        n_steps, n_length = 4, 32
        dataset = dataset.reshape((dataset.shape[0], n_steps, n_length, 2))

        return dataset, labels


def load_session_data(sessionPath):

    data = load_relevant_data_from_file(sessionPath)
    velocities = get_velocity_from_data(data)
    # normalizing data to [0, 1]
    normalized_velocities = normalize_data(velocities, 'builtin')

    return get_partitioned_dataset(normalized_velocities, const.NUM_FEATURES)


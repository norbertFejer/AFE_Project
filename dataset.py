import numpy as np
import csv
import glob
import tensorflow as tf
import random
import os
from sklearn.model_selection import train_test_split

import constants as const
import settings


# load a single file as a numpy array
def load_file(filepath):

    reader = csv.reader(open(filepath, "r"), delimiter=",")
    tmp = list(reader)

    return np.array(tmp).astype('str')


# load a list of files and return as a 3d numpy array
def load_dataset_group(filename, filepath):


    loaded = list()

    # iterating through every file in the given folder
    for file in glob.iglob(filepath + '/' + filename + '/*'):
        data = load_file(file)
        # getting x, y, client timestamp
        relevant_data = get_relevant_data(data)
        velocities = get_velocity_from_data(relevant_data)
        # normalizing data to [0, 1]
        normalized_velocities = normalize_data(velocities, 'builtin')

        if normalized_velocities.shape[1] != 0:
            loaded.extend(normalized_velocities)

    return loaded


# returns the final dataset with the given form (size)
# divides entire array to a given chunk of samples (defined by n_features)
def get_final_dataset(dataset, n_features):

    values = list()

    for i in range( int(len(dataset) / n_features) ):
        int_start = i * n_features
        int_stop = i * n_features + n_features
        values.append(dataset[int_start:int_stop, :].T)

    return np.transpose(np.asarray(values), (0, 2, 1))


# getting: x, y coordinates, client timestamp
def get_relevant_data(data):

    tmp1, tmp2 = [], []
    if settings.selectedDataSet == settings.Dataset.BALABIT:
        tmp1, tmp2 = get_balabit_relevant_data(data)

    if settings.selectedDataSet == settings.Dataset.DFL:
        tmp1, tmp2 = get_dfl_relevant_data(data)

    return np.concatenate((tmp1, tmp2), axis=1).astype(np.float64)


def get_balabit_relevant_data(data):

    tmp1 = data[1:, 4:6]
    tmp2 = data[1:, 1]
    tmp2 = tmp2.reshape(-1, 1)

    return tmp1, tmp2


def get_dfl_relevant_data(data):

    tmp1 = data[1:, 3:5]
    tmp2 = data[1:, 0]
    tmp2 = tmp2.reshape(-1, 1)

    return tmp1, tmp2


# return velocities from data
def get_velocity_from_data(data):

    data = remove_idle_state(data)

    velocity = list()
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
def load_dataset(file, filepath, n_features):

    dataset = np.asarray(load_dataset_group(file, filepath))

    # converting array to n_features number of chunks
    return get_final_dataset(dataset, n_features)


# creating output values

# defines an array with the given size and value
def create_train_label(n_timestamps, value):

    return np.full((n_timestamps, 1), value)


# creating test dataset for given user
# balance is the samples number property, where this value defines the output size
def create_train_input(sessionName, filePath, balance=settings.Balance.POSITIVE):

    # NUM_FEATURES is the feature num in a sample
    dset_positive = load_dataset(sessionName, filePath, const.NUM_FEATURES)
    label_positive = create_train_label(dset_positive.shape[0], 0)  # 0 valid user

    if settings.selectedDataSet == settings.Dataset.DFL:
        dset_positive, X_test, label_positive, y_test = train_test_split(dset_positive, label_positive,
                                                                            test_size=const.TRAIN_SPLIT_VALUE,
                                                                            random_state=const.RANDOM_STATE)
    dset_positive, dset_negative = load_negative_dataset(sessionName, filePath, balance, dset_positive)
    label_negative = create_train_label(dset_negative.shape[0], 1)  # 1 intrusion detected (is illegal)

    X = np.concatenate((dset_positive, dset_negative), axis=0)
    y = np.concatenate((label_positive, label_negative), axis=0)

    return X, y


# creating random input for evaluating the model
# load a list of files and return as a 3d numpy array
def load_random_group(filename, filepath, numSamples):

    loaded = list()
    loadedFileNames = list()

    # getting all subfolders
    directories = os.listdir(filepath)
    # removing from list the user session what we use during training our model
    directories.remove(filename)

    while len(loaded) < numSamples:
        # getting random directory index
        folderInd = random.randint(0, len(directories) - 1)
        # getting all files from given directory
        files = os.listdir(filepath + '/' + directories[folderInd]);

        # getting random file from given folder
        fileInd = random.randint(0, len(files) - 1)

        # checking if we already used this session during training
        # if we used, searching for new one, that had not used before
        while files[fileInd] in loadedFileNames:
            fileInd = random.randint(0, len(files) - 1)

        loadedFileNames.append(files[fileInd])

        # loading file
        data = load_random_file(filepath + '/' + directories[folderInd] + '/' + files[fileInd])

        loaded.extend(data)

    if len(loaded) > numSamples:
        loaded = loaded[:numSamples]

    return loaded


# load given file
def load_random_file(filepath):

    data = load_file(filepath)
    relevant_data = get_relevant_data(data)
    velocities = get_velocity_from_data(relevant_data)
    velocities = normalize_data(velocities, 'builtin')

    return velocities


def load_random_dataset(file, filepath, n_features, n_samples):

    sample_num = n_samples * const.NUM_FEATURES;
    dataset = np.asarray(load_random_group(file, filepath, sample_num))
    dataset = dataset[:sample_num]

    return get_final_dataset(dataset, n_features)


def load_negative_dataset(file, filepath, balance, positive_dataset):

    sample_num = positive_dataset.shape[0]

    negative_dataset = None

    if balance == settings.Balance.POSITIVE:
        # loading sample_num number of random negative session
        negative_dataset = load_random_dataset(file, filepath, const.NUM_FEATURES, sample_num)

    if balance == settings.Balance.NEGATIVE:
        negative_dataset = load_random_dataset(file, filepath, const.NUM_FEATURES,
                                               sample_num + const.NEGATIVE_SAMPLE_NUM)

        # repeating positive dataset values to equalize the negative-positive
        # samples rate
        # this is important for measuring accuracy
        for i in range(const.NEGATIVE_SAMPLE_NUM):
            pos = random.randint(0, sample_num)
            tmp = positive_dataset[pos][np.newaxis, :, :]
            # newaxis is used to increase the dimension of the existing array
            positive_dataset = np.concatenate((positive_dataset, tmp))

    return positive_dataset, negative_dataset


def get_test_data_with_labels(sessionName, testFilesPath, labelsPath, n_features):

    labels = load_file(labelsPath)

    loaded = list()
    output_labels = list()

    finalPath = testFilesPath + sessionName + '/*'

    for file in glob.iglob(finalPath):

        # if given file is labeled
        # (not every test file in given user folder is labeled)
        if file[len(file) - 18: len(file)] in labels:
            data = load_file(file)
            relevant_data = get_relevant_data(data)
            velocities = get_velocity_from_data(relevant_data)
            velocities = normalize_data(velocities, 'builtin')

            label_len = int(np.asarray(velocities).shape[0] / n_features)
            m_len = label_len * n_features
            loaded.extend(np.asarray(velocities)[:m_len, :])

            # given session value
            # 0 is valid user
            # 1 if intrusion detected
            val = np.where(labels == file[len(file) - 18: len(file)])
            output_labels.extend( create_train_label(label_len, int(labels[val[0]][0][1])) )

    return np.asarray(loaded), np.asarray(output_labels)


def load_test_dataset(sessionName, testFilesPath, labelsPath, n_features):

    dataset, labels = get_test_data_with_labels(sessionName, testFilesPath, labelsPath, n_features)

    return get_final_dataset(dataset, n_features), labels


def create_test_dataset(user):

    dset_tmp, labels = [], []

    if settings.selectedDataSet == settings.Dataset.BALABIT:
        dset_tmp, labels = load_test_dataset(user, const.TEST_FILES_PATH, const.TEST_LABELS_PATH, const.NUM_FEATURES)

    if settings.selectedDataSet == settings.Dataset.DFL:
        X, y = load_train_input(user, const.TRAINING_FILES_PATH, settings.balanceType)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=const.TRAIN_SPLIT_VALUE,
                                                            random_state=const.RANDOM_STATE)

        dset_tmp, labels = X_test, y_test

    return get_shaped_dataset(dset_tmp, labels)


# it shapes dataset to given model input dimensions
def get_shaped_dataset(dataset, labels):

    if settings.selectedModel == settings.Model.CNN:
        return dataset, labels

    if settings.selectedModel == settings.Model.TIME_DISTRIBUTED:

        n_steps, n_length = 4, 32
        dataset = dataset.reshape((dataset.shape[0], n_steps, n_length, 2))

        return dataset, labels

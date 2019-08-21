import numpy as np
import pandas as pd
import glob
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import math

# user defined imports
import constants as const
import settings


# ----------------------------------------------------------------------------
# USER AUTHENTICATION

# load a single file from given path
def load_file(filepath):
    return pd.read_csv(filepath).values


# load a single file as a numpy array
# contains only x, y and client timestamps
def load_relevant_data_from_file(filepath):

    if const.SAMPLES_NUM == 'ALL':
        return pd.read_csv(filepath, 
                            usecols=['x', 'y', 'client timestamp'])[['x', 'y', 'client timestamp']].values

    number_of_rows = const.SAMPLES_NUM * const.NUM_FEATURES
    return pd.read_csv(filepath, usecols=['x', 'y', 'client timestamp'],
                        nrows=number_of_rows)[['x', 'y', 'client timestamp']].values


# load a list of files and return as a 3d numpy array
def load_user_sessions(filename, filepath, numberOfSamples):

    loaded = list()

    # iterating through every session file in the given folder
    for session in glob.iglob(filepath + '/' + filename + '/*'):
        data = load_relevant_data_from_file(session)
        velocities = get_velocity_from_data(data)
        # normalizing data to [0, 1]
        normalized_velocities = normalize_data(velocities, 'builtin')

        # checking if normalized_velocities is empty
        if normalized_velocities.shape[1] != 0:
            loaded.extend(normalized_velocities)

            if len(loaded) > numberOfSamples:
                
                return loaded[:numberOfSamples]

    return loaded


# returns the final dataset with the defined shape
# divides entire array to a given chunk of samples (defined by n_features)
def get_partitioned_dataset(dataset, n_features):

    # means that not enough samples is available
    if dataset.shape[0] < n_features:
        return np.empty([1, 0])

    values = list()

    # slicing array to n_feature size of chunks
    for i in range( int(len(dataset) / n_features) ):
        int_start = i * n_features
        int_stop = i * n_features + n_features
        values.append(dataset[int_start:int_stop, :].T)

    return np.transpose(np.asarray(values), (0, 2, 1))


# return velocities from data
def get_velocity_from_data(data):

    data = remove_idle_state(data)

    velocity = list()
    # default period between two consecutive timestamp if dt is 0
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


# searching the maximum and dividing dataset with that value
def normalize_data_user_defined(data):

    data = np.asarray(data)
    max_tuple = np.amax(data, axis=0)
    data[:, 0] /= max_tuple[0]
    data[:, 1] /= max_tuple[1]

    return data


# file - filename
# filepath - the given file path
# n_features - this is the feature number for one chunk
def load_positive_dataset(user, filepath, numberOfRows):

    numberOfSamples = numberOfRows * const.NUM_FEATURES

    dataset = np.asarray(load_user_sessions(user, filepath, numberOfSamples))

    dataset = get_partitioned_dataset(dataset, const.NUM_FEATURES)

    print("Loading positive dataset for user: ", user, " finished.")
    print("Shape before duplication: ", str(dataset.shape))

    # if the positive dataset volume is not enough then replicates the dataset
    if numberOfRows != math.inf and dataset.shape[0] < numberOfRows:

        while dataset.shape[0] < numberOfRows:
            
            dataset = np.concatenate((dataset, dataset), axis=0)

        dataset = dataset[:numberOfRows]

    print("Shape after duplication: ", str(dataset.shape), '\n')

    # converting array to n_features number of chunks
    return dataset


# returns an array with the given size and value
def create_train_label(n_timestamps, value):
    return np.full((n_timestamps, 1), value)


def create_train_dataset(userName, filePath):

    # setting the number of samples to read
    if const.SAMPLES_NUM == 'ALL':
        number_of_samples = math.inf
    else:
        number_of_samples = const.SAMPLES_NUM

    # checking the selected balance type
    if settings.balanceType == settings.Balance.POSITIVE:
         dset_positive, dset_negative = load_positive_balanced(userName, filePath, number_of_samples)

    if settings.balanceType == settings.Balance.NEGATIVE:
        dset_positive, dset_negative = load_negative_balanced(userName, filePath, number_of_samples)

    # checking the train-test split type
    if settings.selectedTrainTestSplitType == settings.TrainTestSplitType.TRAIN_AVAILABLE:
        label_positive = create_train_label(dset_positive.shape[0], 0)  # 0 valid user
        # splitting the given dataset to train and test set
        dset_positive, _, label_positive, _ = train_test_split(dset_positive, 
                                                        label_positive,
                                                        test_size=const.TRAIN_TEST_SPLIT_VALUE,
                                                        random_state=const.RANDOM_STATE)
        dset_negative = dset_negative[:dset_positive.shape[0]]

    if settings.selectedTrainTestSplitType == settings.TrainTestSplitType.TRAIN_TEST_AVAILABLE:
        label_positive = create_train_label(dset_positive.shape[0], 0)  # 0 valid user

    label_negative = create_train_label(dset_negative.shape[0], 1)  # 1 intrusion detected (is illegal)

    print('Final positive dataset shape: ', str(dset_positive.shape))
    print('Final negative dataset shape: ', str(dset_negative.shape), '\n')

    return np.concatenate((dset_positive, dset_negative), axis=0), \
           np.concatenate((label_positive, label_negative), axis=0)


# loads positive balanced dataset, with numberOfRows samples
# reads numberOfRows positive data
# reads numberOfRows negative data, then concatenate these two
# if the number of negative samples is not enough then replicates the missing data
def load_positive_balanced(userName, filePath, numberOfRows):

    dset_positive = load_positive_dataset(userName, filePath, numberOfRows)

    # get the users number from given folder
    numberOfUsers = len( os.listdir(filePath) )
    # defines the negative samples num
    numberOfNegativeRows = int(dset_positive.shape[0] / (numberOfUsers - 1))

    tmp = const.SAMPLES_NUM 
    const.SAMPLES_NUM = numberOfNegativeRows
    dset_negative = load_negative_dataset(userName, filePath, numberOfNegativeRows)
    const.SAMPLES_NUM = tmp

    # if the negative dataset volume is not enough then replicates the dataset
    if dset_negative.shape[0] < dset_positive.shape[0]:

        while dset_negative.shape[0] < dset_positive.shape[0]:
            dset_negative = np.concatenate((dset_negative, dset_negative), axis=0)

        dset_negative = dset_negative[:dset_positive.shape[0]]

    return dset_positive, dset_negative


def load_negative_balanced(userName, filePath, numberOfRows):

    dset_negative = load_negative_dataset(userName, filePath, numberOfRows)

    # it is important to read all possible positive samples
    # without this only reads a part of data, 
    # because of the restriction definied in load_relevant_data_from_file() function 
    # (nrows param)
    tmp = const.SAMPLES_NUM 
    const.SAMPLES_NUM = "ALL"
    dset_positive = load_positive_dataset(userName, filePath, dset_negative.shape[0])
    const.SAMPLES_NUM = tmp

    return dset_positive, dset_negative


def load_negative_dataset(currentUser, filepath, numberOfRows):

    numberOfSamples = numberOfRows * const.NUM_FEATURES
    loadedData = np.empty([0, 128, 2])

    # getting all subfolders
    users = os.listdir(filepath)
    # removing from list the user's session what we use during training our model
    users.remove(currentUser)

    for user in users:
        tmpSessionFileData = load_user_sessions(user, filepath, numberOfSamples)

        if len(tmpSessionFileData) != 0:
            tmpDataset =  get_partitioned_dataset(np.asarray(tmpSessionFileData), const.NUM_FEATURES)

            print("Loading negative dataset from user: ", user, " finished.")
            print("Shape before duplication: ", str(tmpDataset.shape))

            # if the negative dataset volume is not enough then replicates the dataset
            if tmpDataset.shape[0] < numberOfRows:

                while tmpDataset.shape[0] < numberOfRows:
                    tmpDataset = np.concatenate((tmpDataset, tmpDataset), axis=0)

                tmpDataset = tmpDataset[:numberOfRows]

            print("Shape after duplication: ", str(tmpDataset.shape))

            loadedData = np.concatenate((loadedData, tmpDataset), axis=0)

    print('\n')

    return loadedData


def load_random_file_with_velocities(filepath):

    data = load_relevant_data_from_file(filepath)
    velocities = get_velocity_from_data(data)
    velocities = normalize_data(velocities, 'builtin')

    return velocities


def get_action_based_test_data_with_labels(userName, testFilesPath, labelsPath, n_features):

    labels = load_file(labelsPath)

    loadedData = list()
    output_labels = list()

    userSessionsPath = testFilesPath + userName + '/*'

    for sessionName in glob.iglob(userSessionsPath):

        # if given session is labeled
        # (not every test session in given user folder is labeled)
        if sessionName[len(sessionName) - 18: len(sessionName)] in labels:
            tmpData = load_relevant_data_from_file(sessionName)
            velocities = get_velocity_from_data(tmpData)
            velocities = normalize_data(velocities, 'builtin')

            label_len = int(np.asarray(velocities).shape[0] / n_features)
            m_len = label_len * n_features
            loadedData.extend(np.asarray(velocities)[:m_len, :])

            # getting the given session value
            # 0 is valid user
            # 1 if intrusion detected
            val = np.where(labels == sessionName[len(sessionName) - 18: len(sessionName)])
            output_labels.extend(create_train_label(label_len, int(labels[val[0]][0][1])))

    return np.asarray(loadedData), np.asarray(output_labels)


def load_train_test_available_action_based_test_dataset(userName, testFilesPath, labelsPath, n_features):

    dataset, labels = get_action_based_test_data_with_labels(userName, testFilesPath, labelsPath, n_features)

    return get_partitioned_dataset(dataset, n_features), labels


def load_train_available_action_based_test_dataset(userName, filePath):

    # setting the number of samples to read
    if const.SAMPLES_NUM == 'ALL':
        number_of_samples = math.inf
    else:
        number_of_samples = const.SAMPLES_NUM

    if settings.balanceType == settings.Balance.POSITIVE:
        dset_positive, dset_negative = load_positive_balanced(userName, filePath, number_of_samples)

    if settings.balanceType == settings.Balance.NEGATIVE:
        dset_positive, dset_negative = load_negative_balanced(userName, filePath, number_of_samples)

    label_positive = create_train_label(dset_positive.shape[0], 0)  # 0 valid user
    _, dset_positive, _, label_positive = train_test_split(dset_positive, label_positive,
                                                                test_size=const.TRAIN_TEST_SPLIT_VALUE,
                                                                random_state=const.RANDOM_STATE)

    label_negative = create_train_label(dset_negative.shape[0], 1)  # 1 intrusion detected (is illegal)
    _, dset_negative, _, label_negative = train_test_split(dset_negative, label_negative,
                                                                test_size=const.TRAIN_TEST_SPLIT_VALUE,
                                                                random_state=const.RANDOM_STATE)
                                                                
    return np.concatenate((dset_positive, dset_negative), axis=0), \
           np.concatenate((label_positive, label_negative), axis=0)


# returns the formatted dataset for evaluating model
def create_test_dataset(userName):

    dset_tmp, labels = [], []

    # checking if current dataset has separate test set
    if settings.selectedTrainTestSplitType == settings.TrainTestSplitType.TRAIN_TEST_AVAILABLE:
        dset_tmp, labels = load_train_test_available_action_based_test_dataset(userName, 
                                                                                const.TEST_FILES_PATH, 
                                                                                const.TEST_LABELS_PATH, 
                                                                                const.NUM_FEATURES)

    if settings.selectedTrainTestSplitType == settings.TrainTestSplitType.TRAIN_AVAILABLE:
        dset_tmp, labels = load_train_available_action_based_test_dataset(userName, 
                                                                            const.TRAINING_FILES_PATH)

    return get_shaped_dataset(dset_tmp, labels)


# it shapes dataset to given model input dimensions
def get_shaped_dataset(dataset, labels):

    if settings.selectedModel == settings.Model.CNN:
        return dataset, labels

    if settings.selectedModel == settings.Model.TIME_DISTRIBUTED:
        n_steps, n_length = 4, 32
        dataset = dataset.reshape((dataset.shape[0], n_steps, n_length, 2))

        return dataset, labels


# loads given session formated data
def load_session_data(sessionPath):

    data = load_relevant_data_from_file(sessionPath)
    velocities = get_velocity_from_data(data)
    # normalizing data to [0, 1]
    normalized_velocities = normalize_data(velocities, 'builtin')

    return get_partitioned_dataset(normalized_velocities, const.NUM_FEATURES)


# ----------------------------------------------------------------------------
# USER IDENTIFICATION


def get_identification_dataset(method):

    # setting the number of samples to read
    if const.SAMPLES_NUM == 'ALL':
        number_of_samples = math.inf
    else:
        number_of_samples = const.SAMPLES_NUM

    dset_positive = np.empty([0, 128, 2])
    label_positive = np.empty([0, 1])

    for userName in os.listdir(const.TRAINING_FILES_PATH):
        tmp_dset = load_positive_dataset(userName, const.TRAINING_FILES_PATH, number_of_samples)

        userId = int(userName[4:])

        if method == settings.Method.TRAIN:
            tmp_dset, tmp_labels = get_identification_partitioned_train_dataset_with_labels(tmp_dset, userId)

        if method == settings.Method.EVALUATE:
            tmp_dset, tmp_labels = get_identification_partitioned_test_dataset_with_labels(tmp_dset, userId)

        dset_positive = np.concatenate((dset_positive, tmp_dset), axis=0)
        label_positive = np.concatenate((label_positive, tmp_labels), axis=0)

    return dset_positive, label_positive


def get_identification_partitioned_train_dataset_with_labels(dataset, userId):

    labels = create_train_label(dataset.shape[0], userId)  # 0 valid user
    # splitting the given dataset to train and test set
    dataset, _, labels, _ = train_test_split(dataset, 
                                                labels,
                                                test_size=const.TRAIN_TEST_SPLIT_VALUE,
                                                random_state=const.RANDOM_STATE)

    return dataset, labels


def get_identification_partitioned_test_dataset_with_labels(dataset, userId):

    labels = create_train_label(dataset.shape[0], userId)  # 0 valid user
    # splitting the given dataset to train and test set
    _, dataset, _, labels = train_test_split(dataset, 
                                                labels,
                                                test_size=const.TRAIN_TEST_SPLIT_VALUE,
                                                random_state=const.RANDOM_STATE)


    return dataset, labels

def create_identification_train_dataset():
    return get_identification_dataset(settings.Method.TRAIN)


def create_identification_test_dataset():
    return get_identification_dataset(settings.Method.EVALUATE)

from math import inf

import settings
import timeDistributedModel
import cnnModel


# Defines the samples number, that represents a movement
BLOCK_SIZE = 128


# Block size number from given user
# If it's value is math.inf then reads all samples.
# If int value is set, than BLOCK_NUM * BLOCK_SIZE reads rows will be read.
BLOCK_NUM = 400


# Defines the used user name
USER_NAME = 'user35'


# Defines train-test split ratio.
# Only needs if TRAIN_TEST_SPLIT_TYPE is TRAIN_AVAILABLE
# If it's value between (0, 1) then represents the proportion of the dataset to include in the train split.
# If int, represents the absolute number of train samples.
TRAIN_TEST_SPLIT_VALUE = 70


# Defines the batch size
BATCH_SIZE = 32


# Defines the interval when no user interaction occurred.
# It is measured in seconds.
STATELESS_TIME = 2


# Test files location
TEST_FILES_PATH = 'C:/Anaconda projects/Software_mod/MouseDynamics/test_files/'


# Test labels location
TEST_LABELS_PATH = 'C:/Anaconda projects/Software_mod/MouseDynamics/public_labels.csv'


# Training files location
TRAIN_FILES_PATH = 'C:/Anaconda projects/Software_mod/MouseDynamics/training_files'


# Trained models location
TRAINED_MODELS_PATH = 'C:/Anaconda projects/Software_mod/trainedModels'


# Results path
RESULTS_PATH = 'C:/Anaconda projects/Software_mod/evaluationResults'


# Defines random state to initialize environment
RANDOM_STATE = 42


# Set verbose mode on/off
VERBOSE = True


# For DFL dataset we use different settings
if settings.sel_dataset == settings.Dataset.DFL:
    TEST_FILES_PATH = 'C:/Anaconda projects/Software_mod/DFL'
    TRAIN_FILES_PATH = 'C:/Anaconda projects/Software_mod/DFL'
    STATELESS_TIME = 2000

def get_trained_model(trainX, trainy):

    if settings.sel_model == settings.Model.TIME_DISTRIBUTED:

        return timeDistributedModel.train_model(trainX, trainy)

    if settings.sel_model == settings.Model.CNN:

        return cnnModel.train_model(trainX, trainy)
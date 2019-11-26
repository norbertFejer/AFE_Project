from math import inf

import settings
import timeDistributedModel
import cnnModel


# Defines the samples number, that represents a movement
BLOCK_SIZE = 128


# Block size number from given user
# If it's value is math.inf then reads all samples.
# If int value is set, than BLOCK_NUM * BLOCK_SIZE reads rows will be read.
BLOCK_NUM = 300


# Defines the used user name
USER_NAME = 'user7'


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


# Maximum values in pixels
MAX_WIDTH = 4000
MAX_HEIGHT = 4000


# For DFL dataset we use different settings
if settings.sel_dataset == settings.Dataset.DFL:
    TEST_FILES_PATH = 'C:/Anaconda projects/Software_mod/DFL'
    TRAIN_FILES_PATH = 'C:/Anaconda projects/Software_mod/DFL'
    STATELESS_TIME = 2000
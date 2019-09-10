import settings


# number of given chunks to divide
NUM_FEATURES = 128


# samples number from given user
# this value gives NEGATIVE_SAMPLES_NUM * NUM_FEATURES number of samples
# if it's value is 'ALL' then read all samples
# if int value is set, than SAMPLES_NUM * NUM_FEATURES reads
SAMPLES_NUM = 300


# defines the session names
USER_NAME = 'user7'


# defines train-test split ratio
# only needed if TRAIN_TEST_SPLIT_TYPE is TRAIN_AVAILABLE
TRAIN_TEST_SPLIT_VALUE = 0.3


# defines the batch size
BATCH_SIZE = 32


# defines the interval when no user interaction occurred
# it is measured in seconds
STATELESS_TIME = 2


# test files location
TEST_FILES_PATH = 'C:/Anaconda projects/Balabit/MouseDynamics/test_files/'


# test labels location
TEST_LABELS_PATH = 'C:/Anaconda projects/Balabit/MouseDynamics/public_labels.csv'


# training files location
TRAIN_FILES_PATH = 'C:/Anaconda projects/Balabit/MouseDynamics/training_files'


# trained models location
TRAINED_MODELS_PATH = 'C:/Anaconda projects/Balabit/trainedModels'


# results path
RESULTS_PATH = 'C:/Anaconda projects/Balabit/evaluationResults'


# defines random state to initialize environment
RANDOM_STATE = 42


# maximum number of tryings when searching already not used session
MAX_ITER_LOADED_FILES = 25


# for DFL dataset we use different settings
if settings.selectedDataSet == settings.Dataset.DFL:
    TEST_FILES_PATH = 'C:/Anaconda projects/Balabit/DFL'
    TRAIN_FILES_PATH = 'C:/Anaconda projects/Balabit/DFL'
    STATELESS_TIME = 2000
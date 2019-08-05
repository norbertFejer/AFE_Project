import settings

# number of given chunks
NUM_FEATURES = 128

# defines the overplus samples of negative samples
# if balance set to 'negative' then: negative_sample_num = positive_sample_num + NEGATIVE_SAMPLE_NUM
NEGATIVE_SAMPLE_NUM = 1200

# for getting samples
# user name kell legyen
SESSION_NAME = 'user35'

# batch size
BATCH_SIZE = 32

# defines the interval when no user interaction occurred
# in sec
STATELESS_TIME = 2

if settings.selectedDataSet == settings.Dataset.DFL:
    STATELESS_TIME = 2000

# base path
BASE_PATH = 'C:/Anaconda projects/Balabit'

# test files location
TEST_FILES_PATH = BASE_PATH + '/MouseDynamics/test_files/'

if settings.selectedDataSet == settings.Dataset.DFL:
    TEST_FILES_PATH = BASE_PATH

# test files labels
TEST_LABELS_PATH = BASE_PATH + '/MouseDynamics/public_labels.csv'

# training files location
TRAINING_FILES_PATH = 'MouseDynamics/training_files'

if settings.selectedDataSet == settings.Dataset.DFL:
    TRAINING_FILES_PATH = 'C:/Anaconda projects/Balabit/DFL'

# trained models location
TRAINED_MODELS_PATH = 'trainedModels'

# defines random state
RANDOM_STATE = 42

# defines test-split ratio
TRAIN_SPLIT_VALUE = 0.3

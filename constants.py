import settings

# number of given chunks
NUM_FEATURES = 128

# samples number from given user
# this value gives NEGATIVE_SAMPLES_NUM * NUM_FEATURES number of samples
# if it's value is 'ALL' then read all samples
# if int value is set, than SAMPLES_NUM * NUM_FEATURES reads
SAMPLES_NUM = 400

# for getting samples
USER_NAME = 'user35'

# batch size
BATCH_SIZE = 32

# defines the interval when no user interaction occurred
# in sec
STATELESS_TIME = 2

if settings.selectedDataSet == settings.Dataset.DFL:
    STATELESS_TIME = 2000

# test files location
TEST_FILES_PATH = 'C:/Anaconda projects/Balabit/MouseDynamics/test_files/'

if settings.selectedDataSet == settings.Dataset.DFL:
    TEST_FILES_PATH = 'C:/Anaconda projects/Balabit/DFL'

# test files labels
TEST_LABELS_PATH = 'C:/Anaconda projects/Balabit/MouseDynamics/public_labels.csv'

# training files location
TRAINING_FILES_PATH = 'C:/Anaconda projects/Balabit/MouseDynamics/training_files'

if settings.selectedDataSet == settings.Dataset.DFL:
    TRAINING_FILES_PATH = 'C:/Anaconda projects/Balabit/DFL'

# trained models location
TRAINED_MODELS_PATH = 'C:/Anaconda projects/Balabit/trainedModels'

# defines random state
RANDOM_STATE = 42

# defines test-split ratio
TRAIN_SPLIT_VALUE = 0.3

# maximum number of iteration when searching already not used session
MAX_ITER_LOADED_FILES = 25

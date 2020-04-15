import config.settings as stt


""" 
    Global constant values
    ##################################
"""

# Defines the samples number, that represents a movement
BLOCK_SIZE = 128


# Defines the used user name
USER_NAME = 'user35'


# Define the model name to initializing weights for transfer learning
USED_MODEL_FOR_TRANSFER_LEARNING = 'best_identification_Model.CLASSIFIER_FCN_Dataset.DFL_128_2200_trained.hdf5'


# Defines train-test split ratio.
# Only needs if TRAIN_TEST_SPLIT_TYPE is TRAIN_AVAILABLE
# If its value is between (0, 1) then represents the proportion of the dataset to include in the train split.
# If int, represents the absolute number of train samples.
TRAIN_TEST_SPLIT_VALUE = 70


# Defines the batch size
BATCH_SIZE = 32


# Trained models location
TRAINED_MODELS_PATH = 'C:/Anaconda projects/Software_mod/trainedModels'


# Results path
RESULTS_PATH = 'C:/Anaconda projects/Software_mod/evaluationResults'


# Saved images pah
SAVED_IMAGES_PATH = 'C:/Anaconda projects/Software_mod/savedImages'


# Config file location for the automated sript
CONFIG_XML_FILE_LOCATION = './config/config.xml'


# Defines random state to initialize environment
RANDOM_STATE = 42


# Set verbose mode on/off
VERBOSE = True


# Maximum screen sizes in pixels
MAX_WIDTH = 4000
MAX_HEIGHT = 4000


""" 
    Specific constants for each dataset
    ##################################
"""
STATELESS_TIME = 2

if stt.sel_dataset == stt.Dataset.BALABIT:
    # Defines the interval when no user interaction occurred.
    # It is measured in seconds.
    STATELESS_TIME = 2

    # Test files location
    TEST_FILES_PATH = 'C:/Anaconda projects/Software_mod/MouseDynamics/test_files/'

    # Test labels location
    TEST_LABELS_PATH = 'C:/Anaconda projects/Software_mod/MouseDynamics/public_labels.csv'

    # Training files location
    TRAIN_FILES_PATH = 'C:/Anaconda projects/Software_mod/MouseDynamics/training_files'


if stt.sel_dataset == stt.Dataset.DFL:
    TEST_FILES_PATH = 'C:/Anaconda projects/Software_mod/DFL'
    TRAIN_FILES_PATH = 'C:/Anaconda projects/Software_mod/DFL'
    STATELESS_TIME = STATELESS_TIME * 1000


def setter(property_name, arg):
    
    if arg[0].isdigit():
        value = int(arg[0])
    else:
        if arg[0] == 'True' or arg[0] == 'False':
            value = eval(arg[0])
        else:
            value = arg[0]

    globals()[property_name] = value
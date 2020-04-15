from enum import Enum
from math import inf


class Model(Enum):
    CNN = 0
    TIME_DISTRIBUTED = 1
    CLASSIFIER_FCN = 2
    CLASSIFIER_RESNET = 3


class DatasetBalanceType(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'


class Dataset(Enum):
    BALABIT = 0
    DFL = 1


class EvaluationType(Enum):
    SESSION_BASED = 'session_based'
    ACTION_BASED = 'action_based'


class DatasetType(Enum):
    TRAIN_AVAILABLE = 'train_set_available'
    TRAIN_TEST_AVAILABLE = 'train_test_set_available'


class Users(Enum):

    @staticmethod
    def get_balabit_users():
        return ['user7', 'user9', 'user12', 'user15', 'user16', 'user20', 'user21', 'user23', 'user29', 'user35']

    @staticmethod
    def get_dfl_users():
        return ['User1', 'User2', 'User3', 'User4', 'User5', 'User6', 'User7', 'User8', 'User9', 'User10', 
            'User11', 'User12', 'User13', 'User14', 'User15', 'User16', 'User17', 'User18', 'User19',
            'User20', 'User21']


class TrainUserNumber(Enum):
    ALL = 'all'
    SINGLE = 'single'


class EvaluateUserNumber(Enum):
    ALL = 'all'
    SINGLE = 'single'


class Method(Enum):
    TRAIN = 'train_model'
    EVALUATE = 'evaluate_model'
    TRANSFER_LEARNING = 'transfer_learning'


class EvaluationMetric(Enum):
    ALL = 0
    ACC = 1
    AUC = 2
    CONFUSION_MATRIX = 3

class UserRecognitionType(Enum):
    AUTHENTICATION = 'authentication'
    IDENTIFICATION = 'identification'

class ChunkSamplesHandler(Enum):
    CONCATENATE_CHUNKS = 'concatenate'
    DROP_CHUNKS = 'drop'


class ScalingMethod(Enum):
    USER_DEFINED = 0
    MIN_MAX_SCALER = 1
    MAX_ABS_SCALER = 2
    NO_SCALING = 3
    STANDARD_SCALER = 4


class AuthenticationType(Enum):
    BINARY_CLASSIFICATION = 'binary_class'
    ONE_CLASS_CLASSIFICATION = 'one_class'


class RawFeatureType(Enum):
    DX_DY = 'dx_dy'
    VX_VY = 'vx_vy'


class SettingsSource(Enum):
    FROM_PY_FILE = 'from_py_file'
    FROM_XML_FILE = 'from_xml_file'


# Block size number from given user
# If its value is inf then reads all samples.
# If int value is set, then BLOCK_NUM * BLOCK_SIZE rows will be read.
BLOCK_NUM = 300


# Defines the selected method
sel_method = Method.TRAIN


# Defines which model will be used
sel_model = Model.CNN


# Defines used dataset
sel_dataset = Dataset.BALABIT


# Defines the selected recognition type
sel_user_recognition_type = UserRecognitionType.AUTHENTICATION


# Defines the model input type
# VX_VY - horizontal and vertical velocity components
# DX_DY - horizontal and vertical shift components
sel_raw_feature_type = RawFeatureType.VX_VY


# Defines scaling method during creating the training dataset
sel_scaling_method = ScalingMethod.STANDARD_SCALER


# Defines the type of samples negative/positive balance rate
sel_balance_type = DatasetBalanceType.POSITIVE


# It is relevant only for authentication measurement
sel_authentication_type = AuthenticationType.BINARY_CLASSIFICATION


# Defines how we handle the chunk samples
sel_chunck_samples_handler = ChunkSamplesHandler.DROP_CHUNKS


# TRAIN_AVAILABLE means, that we have only train dataset
# TRAIN_TEST_AVAILABLE means, that we have both train and separate test dataset
sel_dataset_type = DatasetType.TRAIN_AVAILABLE


# Defines how many user will be used
# in case of model training
sel_train_user_number = TrainUserNumber.ALL


# It is used for TRANSFER_LEARNING
# If True model weights will be trainable
# If False model weights will be non-trainable
use_trainable_weights_for_transfer_learning = True


# It is used for TRAIN model
# If True and given model already exists the training process will use the pretrained weights
# If False the model weights will be initialized randomly
use_pretrained_weights_for_training_model = False


# Defines how many user will be used 
# in case of model evaluation
sel_evaluate_user_number = EvaluateUserNumber.ALL


# Defines the evaluation metrics
# If we use multiple metrics we have to put into a list, as:  [EvaluationMetric.ACC, EvaluationMetric.AUC, EvaluationMetric.CONFUSION_MATRIX]
# If we use only one metric we define as: EvaluationMetric.ACC
sel_evaluation_metrics = [EvaluationMetric.ACC, EvaluationMetric.AUC, EvaluationMetric.CONFUSION_MATRIX]


# Defines the type of evaluation
sel_evaluation_type = EvaluationType.ACTION_BASED


# Defines saving evaluation results to file
print_evaluation_results_to_file = True


# Defines setting source location
# FROM_PY_FILE means that we use settings from settings.py and constants.py
# FROM_XML_FILE means that we use settings from config.xml
sel_settings_source = SettingsSource.FROM_XML_FILE


def get_balabit_users():
    """ Returns the Balabit Dataset users

        Parameters:
            None

        Returns:
            np.ndarray() - users list
    """ 
    return Users.get_balabit_users()


def get_dfl_users():
    """ Returns the DFL Dataset users

        Parameters:
            None

        Returns:
            np.ndarray() - users list
    """ 
    return Users.get_dfl_users()


def get_users():
    """ Gets users name from the selected dataset

        Parameters:
            None

        Returns:
            np.ndarray() - users list
    """ 
    switcher = { 
        0: get_balabit_users,
        1: get_dfl_users, 
    } 
  
    func = switcher.get(sel_dataset.value, lambda: "Wrong dataset name!")
    return func()


def setter(property_name, args):
    
    if len(args) == 1:

        if args[0].isdigit():
            value = int(args[0])
        else:
            if args[0] == 'inf':
                value = inf
            else:
                value = eval(args[0])

        globals()[property_name] = value

    if len(args) == 2:
        globals()[property_name] = eval(args[0])[args[1]]

    if len(args) > 2:
        tmp = ' '.join(args)
        tmp = tmp[1 : -1]
        tmp = tmp.split(', ')
        globals()[property_name] = []
        for metric in tmp:
            metric = metric.split(' ')
            globals()[property_name].append( eval(metric[0])[metric[1]] )
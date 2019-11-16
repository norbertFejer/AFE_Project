from enum import Enum


class Model(Enum):
    CNN = 'cnnModel'
    TIME_DISTRIBUTED = 'timeDistributedModel'


class DatasetBalanceType(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'


class Dataset(Enum):
    BALABIT = 'balabit'
    DFL = 'dfl'


class EvaluationType(Enum):
    SESSION_BASED = 'session_based'
    ACTION_BASED = 'action_based'


class DatasetType(Enum):
    TRAIN_AVAILABLE = 'trainSetAvailable'
    TRAIN_TEST_AVAILABLE = 'trainTestSetAvailable'


class Users(Enum):

    @staticmethod
    def getBalabitUsers():
        return ['user7', 'user9', 'user12', 'user15', 'user16', 'user20', 'user21', 'user23', 'user29', 'user35']

    @staticmethod
    def getDFLUsers():
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


class EvaluationMetric(Enum):
    ACC = 'accuracy'
    AUC = 'areaUnderCurve'
    CONFUSION_MATRIX = 'confusionMatrix'
    ACC_CONFUSION_MATRIX = 'accAndConfusionMatrix'
    ALL = 'all'

class UserRecognitionType(Enum):
    AUTHENTICATION = 'authentication'
    IDENTIFICATION = 'identification'

class ChunkSamplesHandler(Enum):
    CONCATENATE_CHUNKS = 'concatenate'
    DROP_CHUNKS = 'drop'


class NormalizationMethod(Enum):
    BUILTIN = 'builtin'
    USER_DEFINED = 'userDefined'


# Defines the selected method
sel_method = Method.TRAIN


# Defines which model will be used
sel_model = Model.CNN


# Defines the type of samples negative/positive balance rate
sel_balance_type = DatasetBalanceType.POSITIVE


# Defines used dataset
sel_dataset = Dataset.BALABIT


# Defines the selected recognition type
sel_user_recognition_type = UserRecognitionType.IDENTIFICATION


# Defines what will be with the chunk samples
sel_chunck_samples_handler = ChunkSamplesHandler.CONCATENATE_CHUNKS


# TRAIN_AVAILABLE means, that we have only train dataset
# TRAIN_TEST_AVAILABLE means, that we have both train and separate test dataset
sel_dataset_type = DatasetType.TRAIN_AVAILABLE


# Defines how many user will be used
# in case of model training
sel_train_user_number = TrainUserNumber.SINGLE

# Defines how many user will be used 
# in case of model evaluation
sel_evaluate_user_number = EvaluateUserNumber.SINGLE


# Defines the evaluation metric
sel_evaluation_metric = EvaluationMetric.ACC_CONFUSION_MATRIX


# Defines the type of evaluation
sel_evaluation_type = EvaluationType.ACTION_BASED


# Defines normalizaton method during creating the training dataset
sel_normalization_method = NormalizationMethod.USER_DEFINED
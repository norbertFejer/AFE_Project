from enum import Enum
import timeDistributedModel
import cnnModel


class Model(Enum):
    CNN = 'cnnModel'
    TIME_DISTRIBUTED = 'timeDistributedModel'


class Balance(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'


class Dataset(Enum):
    BALABIT = 'balabit'
    DFL = 'dfl'


class EvaluationType(Enum):
    SESSION_BASED = 'session_based'
    ACTION_BASED = 'action_based'


class TrainTestSplitType(Enum):
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


class EvaluationMetrics(Enum):
    ACC = 'accuracy'
    AUC = 'areaUnderCurve'
    CONFUSION_MATRIX = 'confusionMatrix'
    ACC_CONFUSION_MATRIX = 'accAndConfusionMatrix'
    ALL = 'all'

class UserRecognitionType(Enum):
    AUTHENTICATION = 'authentication'
    IDENTIFICATION = 'identification'

class TestDatasetType(Enum):
    AUGMENTED = 'augmentedDataset'
    DEFAULT = 'defaultDataset'


# defines selected method
selectedMethod = Method.TRAIN


# define which model will be used
selectedModel = Model.CNN


# defines the type of balance
balanceType = Balance.POSITIVE


# defines used dataset
selectedDataSet = Dataset.BALABIT


selectedUserRecognitionType = UserRecognitionType.AUTHENTICATION


# TRAIN_AVAILABLE means, that we have only train dataset
# TRAIN_TEST_AVAILABLE means, that we have both train and test dataset
selectedTrainTestSplitType = TrainTestSplitType.TRAIN_AVAILABLE


# DEFAULT means, that the test dataset contains only the original data
# AUGMENTED means, that the test dataset contains also augmented data
selectedTestDatasetType = TestDatasetType.DEFAULT


# defines how many user dataset will be used
# in case of model training
selectedTrainUserNumber = TrainUserNumber.SINGLE

# defines how many user dataset will be used 
# in case of model evaluation
selectedEvaluateUserNumber = EvaluateUserNumber.SINGLE


# defines the evaluation metric
selectedEvaluationMetric = EvaluationMetrics.ACC_CONFUSION_MATRIX


# defines the type of evaluation
selectedEvaluationType = EvaluationType.ACTION_BASED


# True value will print result to file
saveResultsToFile = True


def get_trained_model(trainX, trainy):

    if selectedModel == Model.TIME_DISTRIBUTED:

        return timeDistributedModel.train_model(trainX, trainy)

    if selectedModel == Model.CNN:

        return cnnModel.train_model(trainX, trainy)

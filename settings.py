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


# define which model will be used
selectedModel = Model.CNN

# defines balance rate
balanceType = Balance.NEGATIVE

# defines used dataset
selectedDataSet = Dataset.BALABIT


def get_trained_model(trainX, trainy):

    if selectedModel == Model.TIME_DISTRIBUTED:

        return timeDistributedModel.train_model(trainX, trainy)

    if selectedModel == Model.CNN:

        return cnnModel.train_model(trainX, trainy)

import settings
import trainModel
import evaluateModel
import constants as const

import os


def train_model_all_user():

    usersArray = []

    if settings.selectedDataSet == settings.Dataset.BALABIT:
        usersArray = settings.Users.getBalabitUsers()

    if settings.selectedDataSet == settings.Dataset.DFL:
        usersArray = settings.Users.getDFLUsers()

    for user in usersArray:
        print('\nTraining model for user: ' + user + '\n')
        trainModel.train_model(user, const.TRAIN_FILES_PATH)
        print('\nTraining model finished for user: ' + user + '\n')


def train_model_single_user():

    trainModel.train_model(const.USER_NAME, const.TRAIN_FILES_PATH)


def evaluate_model_all_user():

    usersArray = []

    if settings.selectedDataSet == settings.Dataset.BALABIT:
        usersArray = settings.Users.getBalabitUsers()

    if settings.selectedDataSet == settings.Dataset.DFL:
        usersArray = settings.Users.getDFLUsers()

    for user in usersArray:
        print('\nEvaluating model for user: ' + user + '\n')
        evaluateModel.evaluate_model(user)
        print('\nEvaluating model finished for user: ' + user + '\n')


def evaluate_model_single_user():

    print('\nEvaluating model for user: ' + const.USER_NAME + '\n')
    evaluateModel.evaluate_model(const.USER_NAME)
    print('\nEvaluating model finished for user: ' + const.USER_NAME + '\n')


def train_model():

    if settings.selectedTrainUserNumber == settings.TrainUserNumber.ALL and \
        settings.selectedUserRecognitionType == settings.UserRecognitionType.AUTHENTICATION:

        train_model_all_user()
        return

    if settings.selectedTrainUserNumber == settings.TrainUserNumber.SINGLE or \
        settings.selectedUserRecognitionType == settings.UserRecognitionType.IDENTIFICATION:

        train_model_single_user()
        return


# checking if saving result to file is set
# initializing result file (.csv)
# creatig evaluationResults folder if not exists
# setting header for file
def initializing_result_file():

    if settings.saveResultsToFile:
        fileTitle = str(settings.selectedDataSet) + '_' +  \
                    str(settings.balanceType) + '_' + \
                    str(const.SAMPLES_NUM) + '_samples.csv'
        fileName = os.path.join(const.RESULTS_PATH, fileTitle)

        # deleting old result
        if os.path.exists(fileName):
            os.remove(fileName)

        # checking if evaluationResult diretory exists
        if not os.path.exists(const.RESULTS_PATH):
            os.makedirs(const.RESULTS_PATH)

        # creating header for file
        headerStr = 'user name,'

        if settings.selectedEvaluationMetric == settings.EvaluationMetrics.ACC:
            headerStr += 'ACC score\n'

        if settings.selectedEvaluationMetric == settings.EvaluationMetrics.AUC:
            headerStr += 'AUC score\n'

        if settings.selectedEvaluationMetric == settings.EvaluationMetrics.ALL:
            headerStr += 'ACC score, AUC score\n'

        file = open(fileName, 'w')
        file.write(headerStr)
        file.close()


def evaluate_model():

    initializing_result_file()

    if settings.selectedEvaluateUserNumber == settings.EvaluateUserNumber.ALL:
        evaluate_model_all_user()
        return

    if settings.selectedEvaluateUserNumber == settings.EvaluateUserNumber.SINGLE:
        evaluate_model_single_user()
        return


def main():

    if settings.selectedMethod == settings.Method.TRAIN:
        train_model()

    if settings.selectedMethod == settings.Method.EVALUATE:
        evaluate_model()


main()
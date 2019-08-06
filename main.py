import settings
import trainModel
import evaluateModel
import constants as const


def train_model_all_user():

    usersArray = []

    if settings.selectedDataSet == settings.Dataset.BALABIT:
        usersArray = settings.Users.getBalabitUsers()

    if settings.selectedDataSet == settings.Dataset.DFL:
        usersArray = settings.Users.getDFLUsers()

    for user in usersArray:
        print('\nTraining model for user: ' + user)
        trainModel.train_model(user, const.TRAINING_FILES_PATH)
        print('\nTraining model finished for user: ' + user)


def train_model_single_user():

    trainModel.train_model(const.USER_NAME, const.TRAINING_FILES_PATH)


def evaluate_model_all_user():

    usersArray = []

    if settings.selectedDataSet == settings.Dataset.BALABIT:
        usersArray = settings.Users.getBalabitUsers()

    if settings.selectedDataSet == settings.Dataset.DFL:
        usersArray = settings.Users.getDFLUsers()

    for user in usersArray:
        print('\nEvaluating model for user: ' + user)
        evaluateModel.evaluate_model(user)
        print('\nEvaluating model finished for user: ' + user)


def evaluate_model_single_user():

    print('\nEvaluating model for user: ' + const.USER_NAME)
    evaluateModel.evaluate_model(const.USER_NAME)
    print('\nEvaluating model finished for user: ' + const.USER_NAME)


def train_model():

    if settings.selectedTrainUserNumber == settings.TrainUserNumber.ALL:
        train_model_all_user()
        return

    if settings.selectedTrainUserNumber == settings.TrainUserNumber.SINGLE:
        train_model_single_user()
        return


def evaluate_model():

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
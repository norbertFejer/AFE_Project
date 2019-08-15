import settings
from keras.models import load_model
import constants as const
import dataset
from keras.utils import to_categorical
import glob
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.metrics import roc_auc_score

def action_based_evaluation(user, testX, testy):

    modelName = const.TRAINED_MODELS_PATH + '/' + str(settings.selectedModel) + '_' + \
                user + '_trained.h5'
    model = load_model(modelName)

    if settings.selectedEvaluationMetric == settings.EvaluationMetrics.ACC:
        print('ACC score: %.2f' % ( get_acc_result(model, testX, testy) ))

    if settings.selectedEvaluationMetric == settings.EvaluationMetrics.AUC:
        print('AUC score: %.2f' % ( get_auc_result(model, testX, testy) ))

    if settings.selectedEvaluationMetric == settings.EvaluationMetrics.ALL:
        print('ACC score: %.2f' % ( get_acc_result(model, testX, testy) ))
        print('AUC score: %.2f' % ( get_auc_result(model, testX, testy) ))



def session_based_evaluation(userName):

    modelName = const.BASE_PATH + '/' + const.TRAINED_MODELS_PATH + '/' + str(settings.selectedModel) + '_' + \
                userName + '_trained.h5'
    model = load_model(modelName)

    labels = dataset.load_file(const.TEST_LABELS_PATH)
    userSessionsPath = const.TEST_FILES_PATH + userName + '/*'

    file = open("test_session_scores.csv", "a+")

    for session in glob.iglob(userSessionsPath):

        # if given file is labeled
        # (not every test file in given user folder is labeled)
        if session[len(session) - 18: len(session)] in labels:
            sessionData = dataset.load_session_data(session)

            if sessionData.shape[1] == 0:
                continue

            # evaluate model
            predictions = model.predict(sessionData)

            labelValue = np.where(labels == session[len(session) - 18: len(session)])

            val = 1 - labels[labelValue[0]][0][1]

            sum = 0
            for i in range(len(predictions)):
                sum += predictions[i, 0]

            avg = sum / len(predictions)
            file.write(str(val) + "," + str(avg) + "\n")

    file.close()


# computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
def get_auc_result(model, testX, y_true):

    y_scores = model.predict(testX)
    
    return roc_auc_score(y_true, y_scores)


# computes Accuracy
def get_acc_result(model, testX, y_true):

    # evaluate model
    _, accuracy = model.evaluate(testX, y_true, batch_size=const.BATCH_SIZE, verbose=1)
    
    return accuracy


def evaluate_model(userName):

    if settings.selectedEvaluationType == settings.EvaluationType.ACTION_BASED:

        print('Loading test dataset...')
        testX, testy = dataset.create_test_dataset(userName)

        testy = to_categorical(testy)
        print('Loading test dataset finished')
        print(testX.shape)

        action_based_evaluation(userName, testX, testy)

    if settings.selectedEvaluationType == settings.EvaluationType.SESSION_BASED:

        session_based_evaluation(userName)


def plotScores(scorefilename, title='Title'):

    data = pd.read_csv(scorefilename, names=['label', 'score'])
    df = pd.DataFrame(data)
    positive = df.query('label==1')
    negative = df.query('label==0')
    positive_scores = positive['score']
    negative_scores = negative['score']

    min_value = 1
    max_value = 100

    # bins = np.linspace(min_value, max_value, 100)
    bins = 100

    plt.hist(negative_scores, bins, alpha=0.5, color='red')
    plt.hist(positive_scores, bins, alpha=0.5, color='green')
    plt.title(title)
    plt.xlabel('Score value')
    plt.ylabel('#Occurences')
    red_patch = mpatches.Patch(color='red', label='Negative')
    green_patch = mpatches.Patch(color='green', label='Positive')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    return


def plotROC(scorefilename, title='ROC'):

    data = pd.read_csv(scorefilename, names=['label', 'score'])
    df = pd.DataFrame(data)
    y_true = df.label
    y_scores = df.score
    sum = np.sum(y_true, axis=0)
    num_positives = sum
    num_negatives = df.shape[0] - num_positives
    print("num positives: " + str(num_positives) + "\tnum_negatives:"+str(num_negatives) )
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_scores)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve(area= % 0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# plotScores('test_session_scores.csv','Scores')
# plotROC('test_session_scores.csv','ROC-AUC')
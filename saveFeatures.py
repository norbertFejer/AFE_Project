import constants as const
import settings
import dataset
from keras.models import load_model
from keras.models import Model
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def save_model_features(testX, userId):

    modelName = const.BASE_PATH + '/' + const.TRAINED_MODELS_PATH + '/' + str(settings.selectedModel) + \
                '_user' + str(userId) + '_trained.h5'
    model = load_model(modelName)

    layer_name = 'features_layer'
    tmp_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = tmp_model.predict(testX)

    with open("features.csv", "a+", newline='') as f:
        writer = csv.writer(f, delimiter=',')

        for row in intermediate_output:
            tmp_row = row.reshape((1, 60))
            writer.writerow(np.append(tmp_row, [userId]))


def save_features():

    userId_list = [7, 9, 12, 15, 16, 20, 21, 23, 29, 35]

    print("Writing to CSV file....")

    for userId in userId_list:
        print('Processing user' + str(userId) + '...')
        # trainX, trainy = dataset.create_train_input('user' + str(userId), const.TRAINING_FILES_PATH,
        #                                                   settings.balanceType)
        testX, testy = dataset.create_test_dataset('user' + str(userId))
        # it is important to reshape dataset for timeDistributed model
        # n_steps, n_length = 4, 32
        # testX = testX.reshape((testX.shape[0], n_steps, n_length, 2))
        print(testX.shape)
        save_model_features(testX, userId)

    print("Finished writing to CSV file....")


def load_features(fileName):
    data = np.genfromtxt(fileName, delimiter=',')
    return data


def clustering_data():
    features = load_features('timeDistributed_features.csv')

    kmeans = KMeans(n_clusters=10, random_state=0).fit(features[:, :60])

    y_true = features[:, 60:61]
    y_pred = kmeans.labels_
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8', 'class 9', 'class 10']
    print(classification_report(y_true, y_pred, target_names=target_names))

def run():
    features = load_features('timeDistributed_features.csv')

    X = features[:, :60]
    y = features[:, -1]

    model = RandomForestClassifier(n_estimators=100,
                                   random_state=0)
    scoring = ['accuracy']
    num_folds = 3
    scores = cross_validate(model, X, y, scoring=scoring, cv=num_folds)

    print(str(num_folds) + '-fold cross-validation results: ')
    for i in range(0, num_folds):
        print('\tFold ' + str(i + 1) + ':' + str(scores['test_accuracy'][i]))

    print('Avearage[accuracy]: ' + str(np.average(scores['test_accuracy'])))
    print('Std[accuracy]: ' + str(np.std(scores['test_accuracy'])))


save_features()
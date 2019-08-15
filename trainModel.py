import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
import os

import settings
import dataset
import constants as const
from keras.utils import to_categorical
import matplotlib.pyplot as plt

np.random.seed(42)

rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"]="0" #specific index
# ---------------------------------------------------------------------------------


def train_model(user, trainingFilesPath):

    # load input dataset
    print('Loading train dataset...')
    trainX, trainy = dataset.create_train_dataset(user, trainingFilesPath)

    trainy = to_categorical(trainy)
    print('Loading train dataset finished')
    print(trainX.shape)

    model, history = settings.get_trained_model(trainX, trainy)

    # saving the trained model
    modelName = str(settings.selectedModel) + '_' + user + '_trained.h5'

    if not os.path.exists(const.TRAINED_MODELS_PATH):
        os.makedirs(const.TRAINED_MODELS_PATH)

    model.save(const.TRAINED_MODELS_PATH + '/' + modelName)

    return history


def plot_train():

    history = train_model()

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.title('Model training')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Acc', 'Loss'], loc='upper left')
    plt.show()

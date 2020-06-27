import os
import csv
from numpy import genfromtxt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.svm import OneClassSVM
from sklearn import metrics

import config.constants as const
import utils.dataVisualization as dvisuals


# Global parameters
results = {}
dataset_path = 'C:/Anaconda projects/Software_mod/sapimouse_dataset'
saved_images_path = 'C:/Anaconda projects/Software_mod/analysis/saved_images'
stateless_time = 1000
# 0 variable length blocks
# > 0 fixed length blocks
BLOCK_SIZE = 128
BLOCK_NUM = 200
TRAIN = 'train'
TEST = 'test'
AGGREGATE_BLOCK_NUM = 1


def print_dataset_statistics():

    f = open('analysis/statistics.csv', "w")
    f.write('username, session_1min, session_2min\n')
    for username in os.listdir(dataset_path):
        f.write(username + ',')

        for session_name in os.listdir(dataset_path + '/' + username):
            df = pd.read_csv(dataset_path + '/' + username + '/' + session_name)
            df = df.drop_duplicates()
            df = df.reset_index()
            f.write(str(df.shape[0]) + ',')

        f.write('\n')

    f.close()


def plot_session_mouse_movements(username, session_name):

    session_path = dataset_path + '/' + username + '/' + session_name
    df = pd.read_csv(session_path)

    df = df.drop_duplicates()
    df = df.reset_index()

    start_pos = 0
    end_pos = 1
    actual_state = 'Released'

    while end_pos < df.shape[0] and start_pos <= end_pos:

        if actual_state == 'Released':
            actual_state = 'Pressed'

            while (df['state'][end_pos] != 'Pressed' and end_pos < df.shape[0] - 1) and \
                    (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):
                end_pos = end_pos + 1
        
        else:
            actual_state = 'Released'

            while df['state'][end_pos] != 'Released' and end_pos < df.shape[0] - 1 and \
                    (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):   
                end_pos = end_pos + 1

        # ignoring double click
        if end_pos - start_pos > 1:
            if df['state'][end_pos - 1] == 'Drag':
                #plt.plot(df['x'][start_pos:end_pos + 1], df['y'][start_pos:end_pos + 1], marker='o')
                print(start_pos + 2, '-', end_pos + 3)
            else:
                #plt.plot(df['x'][start_pos + 1:end_pos], df['y'][start_pos + 1:end_pos], marker='o')
                print(start_pos + 3, '-', end_pos + 2)
            
            # plt.xlim((0, df['x'].max() + 10))
            # plt.ylim((0, df['y'].max() + 10))
            # plt.tight_layout()
            # plt.show()
        
        start_pos = end_pos
        end_pos = end_pos + 1


    if not os.path.exists(saved_images_path + '/' + username):
        os.makedirs(saved_images_path + '/' + username)

    #plt.tight_layout()
    # plt.show()
    #plt.savefig(saved_images_path + '/' + username + '/' + session_name + '.png', dpi = 150, bbox_inches='tight')


def plot_all_user_mouse_movements():

    for username in os.listdir(dataset_path):
        print("Ploting mouse movements for user: ", username)

        for session_name in os.listdir(dataset_path + '/' + username):
            plot_session_mouse_movements(username, session_name)


def print_user_blocks(username, session_name):
    session_path = dataset_path + '/' + username + '/' + session_name
    df = pd.read_csv(session_path)

    df = df.drop_duplicates()
    df = df.reset_index()

    start_pos = 0
    end_pos = 1
    actual_state = 'Released'
    block_num = 0
    block_sum = 0

    if not os.path.exists(saved_images_path + '/' + username):
        os.makedirs(saved_images_path + '/' + username)

    f = open(saved_images_path + '/' + username + '/' + session_name + '.csv', "w")
    f.write('boundary, block size\n')

    while end_pos < df.shape[0] and start_pos <= end_pos:

        if actual_state == 'Released':
            actual_state = 'Pressed'

            while (df['state'][end_pos] != 'Pressed' and end_pos < df.shape[0] - 1) and \
                    (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):
                end_pos = end_pos + 1
        
        else:
            actual_state = 'Released'

            while df['state'][end_pos] != 'Released' and end_pos < df.shape[0] - 1 and \
                    (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):   
                end_pos = end_pos + 1

        # ignoring double click
        if end_pos - start_pos > 1:
            if df['state'][end_pos - 1] == 'Drag':
                #print(start_pos + 2, '-', end_pos + 3)
                f.write(str(start_pos + 2) + '-' + str(end_pos + 3) + ',' + str(end_pos - start_pos - 1) + '\n')
            else:
                #print(start_pos + 3, '-', end_pos + 2)
                f.write(str(start_pos + 3) + '-' + str(end_pos + 2) + ',' + str(end_pos - start_pos - 1) + '\n')
        
        block_num = block_num + 1
        block_sum = block_sum + (end_pos - start_pos - 1)

        start_pos = end_pos
        end_pos = end_pos + 1

    f.close()
    return block_sum, block_num


def print_all_user_blocks_sizes():

    f = open('C:/Anaconda projects/Software_mod/analysis/block_statistics.csv', "w")
    f.write('username, avg block size 1min, block_num 1min, avg block size 3m, block_num 3m\n')

    for username in os.listdir(dataset_path):
        print('Printing blocks for user: ', username)

        f.write(username)

        for session_name in os.listdir(dataset_path + '/' + username):
            block_sum, block_num = print_user_blocks(username, session_name)

            f.write(',' + str(block_sum / block_num) + ',' + str(block_num))

        f.write('\n')

    f.close()


def print_user_characteristics(username, session_name):
    session_path = dataset_path + '/' + username + '/' + session_name
    df = pd.read_csv(session_path)

    df = df.drop_duplicates()
    df = df.reset_index()

    start_pos = 0
    end_pos = 1
    actual_state = 'Released'

    filtered_df = pd.DataFrame(columns=['client timestamp', 'x', 'y'])

    while end_pos < df.shape[0] and start_pos <= end_pos:

        if actual_state == 'Released':
            actual_state = 'Pressed'

            while (df['state'][end_pos] != 'Pressed' and end_pos < df.shape[0] - 1) and \
                    (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):
                end_pos = end_pos + 1
        
        else:
            actual_state = 'Released'

            while df['state'][end_pos] != 'Released' and end_pos < df.shape[0] - 1 and \
                    (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):   
                end_pos = end_pos + 1

        # ignoring double click
        if end_pos - start_pos > 1:

            if df['state'][end_pos - 1] == 'Drag':
                #print(start_pos + 2, '-', end_pos + 3)
                filtered_df = pd.concat([filtered_df, df[['client timestamp', 'x', 'y']][start_pos:end_pos + 1]])
            else:
                #print(start_pos + 3, '-', end_pos + 2)
                filtered_df = pd.concat([filtered_df, df[['client timestamp', 'x', 'y']][start_pos + 1:end_pos]])
        

        start_pos = end_pos
        end_pos = end_pos + 1

    filtered_df = filtered_df.diff()
    filtered_df = filtered_df.rename(columns={'client timestamp': 'dt', 'x': 'dx', 'y': 'dy'})
    filtered_df['dt'] = filtered_df['dt'] / 1000
    filtered_df = filtered_df.drop(filtered_df[filtered_df['dt'] == 0].index)

    filtered_df['vx'] = filtered_df['dx'] / filtered_df['dt']
    filtered_df['vy'] = filtered_df['dy'] / filtered_df['dt']

    filtered_df[1:].to_csv(saved_images_path + '/' + username + '/' + session_name + '_calc.csv')


def print_all_user_characteristics():

    for username in os.listdir(dataset_path):
        print('Printing characteristics for user: ', username)

        for session_name in os.listdir(dataset_path + '/' + username):
            print_user_characteristics(username, session_name)


def get_mouse_movements(username, session_name):
    session_path = dataset_path + '/' + username + '/' + session_name
    df = pd.read_csv(session_path)

    df = df.drop_duplicates()
    df = df.reset_index()

    start_pos = 0
    end_pos = 1
    actual_state = 'Released'

    filtered_df = pd.DataFrame(columns=['client timestamp', 'x', 'y'])
    while end_pos < df.shape[0] and start_pos <= end_pos:

        if actual_state == 'Released':
            state_boundary = 'Pressed'
        else:
            state_boundary = 'Released'

        # getting the mouse movement boundary (pressed or released)
        while df['state'][end_pos] != state_boundary and end_pos < df.shape[0] - 1 and \
                (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):
            end_pos = end_pos + 1
        
        # if the boundary occured do to stateless
        # and this state is not overlap with the real boundary (pressed or released)
        # otherwise we do not change actual_state variable
        if (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time) or \
                (df['state'][end_pos] != 'Pressed' and df['state'][end_pos] != 'Released'):

            if actual_state == 'Released':
                actual_state = 'Pressed'
            else:
                actual_state = 'Released'

        # ignoring double click
        if end_pos - start_pos > 1:
            if df['state'][end_pos - 1] == 'Drag':
                #print(start_pos + 2, '-', end_pos + 3)
                filtered_df = pd.concat([filtered_df, df[['client timestamp', 'x', 'y']][start_pos:end_pos + 1]])
            else:
                #print(start_pos + 3, '-', end_pos + 2)
                filtered_df = pd.concat([filtered_df, df[['client timestamp', 'x', 'y']][start_pos + 1:end_pos]])

        start_pos = end_pos
        end_pos = end_pos + 1

    return filtered_df

###################################################################################################################################33
# Generating mouse movements for training the model

def get_variable_length_mouse_movement(username, session_name):
    session_path = dataset_path + '/' + username + '/' + session_name
    df = pd.read_csv(session_path)

    df = df.drop_duplicates()
    df = df.reset_index()

    start_pos = 0
    end_pos = 1
    actual_state = 'Released'

    data_list = []
    while end_pos < df.shape[0] and start_pos <= end_pos:

        if actual_state == 'Released':
            state_boundary = 'Pressed'
        else:
            state_boundary = 'Released'

        # getting the mouse movement boundary (pressed or released)
        while df['state'][end_pos] != state_boundary and end_pos < df.shape[0] - 1 and \
                (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):
            end_pos = end_pos + 1
        
        # if the boundary occured do to stateless
        # and this state is not overlap with the real boundary (pressed or released)
        # otherwise we do not change actual_state variable
        if (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time) or \
                (df['state'][end_pos] != 'Pressed' and df['state'][end_pos] != 'Released'):

            if actual_state == 'Released':
                actual_state = 'Pressed'
            else:
                actual_state = 'Released'

        # ignoring double click
        if end_pos - start_pos > 1:
            if df['state'][end_pos - 1] == 'Drag':
                #print(start_pos + 2, '-', end_pos + 3)
                head = start_pos
                tail = end_pos + 1
            else:
                #print(start_pos + 3, '-', end_pos + 2)
                head = start_pos + 1
                tail = end_pos

            tmp_df = df[['client timestamp', 'x', 'y']][head:tail]
            tmp_df = tmp_df.diff()
            tmp_df.loc[ tmp_df['client timestamp'] <= 0.01, 'client timestamp' ] = 0.016
            tmp_df['vx'] = tmp_df['x'] / tmp_df['client timestamp']
            tmp_df['vy'] = tmp_df['y'] / tmp_df['client timestamp']
            data_list.append(tmp_df[['vx', 'vy']][1:].to_numpy())

        start_pos = end_pos
        end_pos = end_pos + 1

    return np.array([np.array(xi) for xi in data_list])


def get_fixed_length_mouse_movement(username, session_name, method):
    session_path = dataset_path + '/' + username + '/' + session_name

    if method == TRAIN:
        df = pd.read_csv(session_path)
    else:
        max_row = BLOCK_NUM * BLOCK_SIZE + 1500
        df = pd.read_csv(session_path, nrows=max_row)

    df = df.drop_duplicates().reset_index()

    start_pos = 0
    end_pos = 1
    actual_state = 'Released'

    filtered_df = pd.DataFrame(columns=['vx', 'vy'])
    while end_pos < df.shape[0] and start_pos <= end_pos:

        if actual_state == 'Released':
            state_boundary = 'Pressed'
        else:
            state_boundary = 'Released'

        # getting the mouse movement boundary (pressed or released)
        while df['state'][end_pos] != state_boundary and end_pos < df.shape[0] - 1 and \
                (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time):
            end_pos = end_pos + 1
        
        # if the boundary occured do to stateless
        # and this state is not overlap with the real boundary (pressed or released)
        # otherwise we do not change actual_state variable
        if (df['client timestamp'][end_pos] - df['client timestamp'][end_pos - 1] <= stateless_time) or \
                (df['state'][end_pos] != 'Pressed' and df['state'][end_pos] != 'Released'):

            if actual_state == 'Released':
                actual_state = 'Pressed'
            else:
                actual_state = 'Released'

        # ignoring double click
        if end_pos - start_pos > 1:
            if df['state'][end_pos - 1] == 'Drag':
                #print(start_pos + 2, '-', end_pos + 3)
                head = start_pos
                tail = end_pos + 1
            else:
                #print(start_pos + 3, '-', end_pos + 2)
                head = start_pos + 1
                tail = end_pos

            tmp_df = df[['client timestamp', 'x', 'y']][head : tail].diff().abs()
            tmp_df = tmp_df.drop(tmp_df[tmp_df['client timestamp'] < 0.001].index)
            tmp_df['vx'] = tmp_df['x'] / tmp_df['client timestamp']
            tmp_df['vy'] = tmp_df['y'] / tmp_df['client timestamp']
            filtered_df = pd.concat([filtered_df, tmp_df[['vx', 'vy']][1:]])

        if filtered_df.shape[0] > BLOCK_NUM * BLOCK_SIZE:
            break

        start_pos = end_pos
        end_pos = end_pos + 1

    #filtered_df.to_csv(username + '_out.csv')
    filtered_df = filtered_df.to_numpy()
    row_num = int(filtered_df.shape[0] / BLOCK_SIZE)

    if BLOCK_NUM < row_num:
        row_num = BLOCK_NUM

    filtered_df = filtered_df[:row_num * BLOCK_SIZE]

    return np.reshape(filtered_df, (row_num, BLOCK_SIZE, 2))


def print_velocities_to_csv(username, session_name, file_path):

    data = get_fixed_length_mouse_movement(username, session_name)

    f = open(file_path + '_' + str(BLOCK_SIZE) + '_blocks.csv', 'w')
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            f.write(str(data[i][j][0]) + ',')

        for j in range(data[i].shape[0]):
            f.write(str(data[i][j][1]) + ',')

        f.write(username[5:])
        f.write('\n')

    f.close()



def get_model1():

    inputs = keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(input_dim=128, output_dim=16, mask_zero=True)(inputs)
    outputs = layers.LSTM(32)(x)

    model = keras.Model(inputs, outputs)

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    return model


def get_model():

    input_layer = keras.layers.Input(shape=(input_shape[1], input_shape[2]))

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(2, activation='softmax')(gap_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['binary_accuracy'])

    return model


def train_model():

    file_path = 'C:/Anaconda projects/Software_mod/analysis/saved_images/user016/session_2020_05_25_1min_variable_mouse_movements.csv'

    data_list = []
    with open(file_path, 'r') as f:
        
        for line in f:
            tmp_data =  line.strip().split(',')
            while("" in tmp_data) : 
                tmp_data.remove("") 

            tmp_data = [float(i) for i in tmp_data]
            data_list.append( tmp_data )

    #trainX = np.array([np.array(xi) for xi in data_list])
    trainy = np.full(len(data_list), 0)

    #print(trainX.shape)
    print(trainy.shape)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_list, padding="post")

    model = get_model()
    model.fit(padded_inputs, trainy, epochs=20, verbose=1)


def create_train_dataset(username):

    print('Creating train dataset...')

    for session_name in os.listdir(dataset_path + '/' + username):
        if session_name[-8:] == '3min.csv':
            data = get_fixed_length_mouse_movement(username, session_name, TRAIN)

    return scale_dataset( data )



def create_test_dataset(username):

    print('Creating test dataset...')

    for session_name in os.listdir(dataset_path + '/' + username):
        if session_name[-8:] == '1min.csv':
            data = get_fixed_length_mouse_movement(username, session_name, TEST)

    y_true = np.full(data.shape[0], 0)
    pos_samples_num = y_true.shape[0]

    user_list = os.listdir(dataset_path)
    user_list.remove(username)

    for usern in user_list:

        session_list = os.listdir(dataset_path + '/' + usern)
        for session_name in session_list:

            if session_name[-8:] == '1min.csv':
                
                tmp_data = get_fixed_length_mouse_movement(usern, session_name, TEST)
                data = np.concatenate((data, tmp_data), axis=0)
                y_true = np.concatenate((y_true, np.full(tmp_data.shape[0], 1)), axis=0)

    return scale_dataset(data), y_true, pos_samples_num


def get_model_output_features(train_data, test_data):
    
    #model_name = 'best_identification_Model.CLASSIFIER_FCN_Dataset.DFL_' + str(BLOCK_SIZE) + '_1000_trained.hdf5'
    model_name = 'dfl_1000_ResNet.hdf5'
    print('Loaded model for feature extraction:', model_name)
    model_path = const.TRAINED_MODELS_PATH + '/' + model_name

    model = load_model(model_path)
    model._layers.pop()
    model.outputs = [model.layers[-1].output]

    return model.predict(train_data), model.predict(test_data)


def train_test_classifier(user, trainX, testX, testy, pos_samples_num):
    global AGGREGATE_BLOCK_NUM

    print('\nTraining model for user:', user, '...')
    # Fit selected network
    classifier = OneClassSVM(kernel='rbf', gamma='scale', verbose=True).fit(trainX)

    print('\nEvaluating model...')
    # Getting AUC from predicted values
    results[user] = get_auc_result(classifier, testX, testy, pos_samples_num)
    print('Evaluated AUC value for user:', user, 'is', results[user], '\n')

    keras.backend.clear_session()


def print_result_to_file():
    """ Save evaluation results to file

        Parameters:
            file_name (str) - filename

        Returns:
            None
    """
    file_name = 'C:/Anaconda projects/Software_mod/analysis/ocsvm_results_' + str(BLOCK_SIZE) + '_block.csv' 
    file = open(file_name, 'w')
    file.write('username,AUC\n')
    
    # Iterating through each user's AUC values
    for user, value in results.items():
        file.write(str(user) + ',' + str(value) + '\n')

    file.close()


def aggregate_blocks(y_pred, pos_samples_num):
    """ Aggregate blocks for evaluating model using multiple blocks

        Parameters:
            y_pred (np.ndarray) - predicted result for each block

        Returns:
            None
    """
    
    if AGGREGATE_BLOCK_NUM == 1:
        return y_pred

    y_pred = y_pred.astype(float)
    # Aggregating positive class values
    for i in range(pos_samples_num - AGGREGATE_BLOCK_NUM + 1):
        y_pred[i] = np.average(y_pred[i : i + AGGREGATE_BLOCK_NUM], axis=0)

    # Aggregating negative class values
    for i in range(pos_samples_num, len(y_pred) - AGGREGATE_BLOCK_NUM + 1):
        y_pred[i] = np.average(y_pred[i : i + AGGREGATE_BLOCK_NUM], axis=0)

    return y_pred


def get_auc_result(classifier, testX, y_true, pos_samples_num):

    y_pred = classifier.score_samples(testX)
    y_pred = aggregate_blocks(y_pred, pos_samples_num)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=0)

    return metrics.auc(fpr, tpr)


def scale_dataset(data):

    for i in range(data.shape[0]):
        data[i] = standard_scaler(data[i])

    return data


def standard_scaler(data):

    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)

    if std_val[0] == 0:
        std_val[0] = 0.001
    if std_val[1] == 0:
        std_val[1] = 0.001

    data = (data - mean_val) / std_val

    return data


def fit_and_evaluate_network(username):
    global BLOCK_NUM, BLOCK_SIZE

    BLOCK_NUM = 55
    trainX = create_train_dataset(username)
    
    BLOCK_NUM = 15
    testX, testy, pos_samples_num = create_test_dataset(username)

    print('Getting features from model...')
    train_features, test_features = get_model_output_features(trainX, testX)

    print('Train dataset shape:', train_features.shape)
    print('Test dataset shape:', test_features.shape)

    train_test_classifier(username, train_features, test_features, testy, pos_samples_num)


def fit_and_evaluate_all_user():

    id = 0
    for username in os.listdir(dataset_path):

        if id < 10:
            fit_and_evaluate_network(username)
        else:
            break
        id = id + 1

    print_result_to_file()
    dvisuals.plot_occ_results_boxplot(results, 'SapiMouse')


def print_all_user_raw_data():

    file_path = 'C:/Anaconda projects/Software_mod/analysis/raw_data'

    for username in os.listdir(dataset_path):
        print('Printing velocities for user: ', username)

        if not os.path.exists(file_path + '/' + username):
            os.makedirs(file_path + '/' + username)

        for session_name in os.listdir(dataset_path + '/' + username):
            file_name = file_path + '/' + username + '/' + session_name[:-4]
            print_velocities_to_csv(username, session_name, file_name)


def create_labels(length, value):
    return np.full((length, 1), value)


def get_identification_dataset():

    data = np.empty(shape=(0, 128, 2))
    labels = np.ndarray(shape=(0, 1), dtype=int)

    id = 0
    for username in os.listdir(dataset_path):

        #if id > 39:
        
        BLOCK_NUM = 1000
        tmp_data = np.empty(shape=(0, 128, 2))

        for session_name in os.listdir(dataset_path + '/' + username):
            tmp_data = np.concatenate((tmp_data, get_fixed_length_mouse_movement(username, session_name, TRAIN) ), axis=0)

        BLOCK_NUM = 1000
        tmp_data = tmp_data[:BLOCK_NUM]
        print('Dataset shape for user: ', username, 'is ', tmp_data.shape[0])

        data = np.concatenate((data, tmp_data ), axis=0)
        labels = np.concatenate((labels, create_labels(tmp_data.shape[0], id)), axis=0)

        id = id + 1
        if id > 42:
            break

    return data, labels


def plot_datasets_distributions():
    df = pd.read_csv('C:/Anaconda projects/Software_mod/analysis/distribution.csv')

    # color = np.array(["green", "green", "green", "green", "green", "green", "green", "green", "green","green",
    #         "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", 
    #         "orangered", "orangered", "orangered", "orangered", "orangered", "orangered", "orangered", "orangered", "orangered", "orangered"])
    # z = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # plt.scatter(df['id'], df['axes'], s=df['size'], c=color[z], alpha=0.6)

    #plt.bar(df['id'], df['size'])

    #plt.show()

def test():
    N = 10
    dfl = (315547,191106,116281,110565,82175,40765,29804,29024,28373,24672)
    balabit = (2512,2483,1348,945,540,519,459,420,400,289)
    sapimouse = (171,158,158,152,150,140,122,103,98,85)

    ind = np.arange(N)    # the x locations for the groups
    #width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, dfl, edgecolor='black', linewidth=2)
    p2 = plt.bar(ind, balabit, edgecolor='black', linewidth=2)
    p3 = plt.bar(ind, sapimouse, edgecolor='black', linewidth=2)

    plt.ylabel('Egérmozgások száma (128 egéresemény)', fontsize=30)
    plt.title('Az adathalmazok számossága', fontsize=30)
    plt.xticks(ind, ('user#1', 'user#2', 'user#3', 'user#4', 'user#5', 'user#6', 'user#7', 'user#8', 'user#9', 'user#10'), fontsize=30, rotation=15)
    plt.yticks(fontsize=30)
    #plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0], p3[0]), ('DFL', 'Balabit', 'SapiMouse'), fontsize=26)

    plt.show()



if __name__ == "__main__":
    #print_dataset_statistics()
    #plot_session_mouse_movements('user022', 'session_2020_05_21_1min.csv')
    #plot_all_user_mouse_movements()
    #print_all_user_blocks_sizes()
    #print_user_characteristics('user016', 'session_2020_05_25_1min.csv')
    #print_all_user_characteristics()
    #tmp_list = get_variable_length_mouse_movement('user016', 'session_2020_05_25_1min.csv')
    #print_variable_length_mouse_movements('user016', 'session_2020_05_25_1min.csv')
    #train_model()
    #df = get_fixed_length_mouse_movement('user065', 'session_2020_06_09_3min.csv')
    #print(df.shape)
    #df = get_variable_length_mouse_movement('user016', 'session_2020_05_25_1min.csv')
    #print_velocities_to_csv('user016', 'session_2020_05_25_1min.csv')
    #arr = get_fixed_length_mouse_movement('user016', 'session_2020_05_25_1min.csv')
    #print(arr[0][0][1])
    #data, y = create_test_dataset('user016')
    #print(data.shape)
    #data = create_train_dataset('user041')
    #print(data.shape)
    #features = get_model_output_features(data)
    #fit_and_evaluate_network('user016')
    #dataset, y = create_test_dataset('user016')
    #print(dataset.shape)
    fit_and_evaluate_all_user()
    #print_all_user_raw_data()
    #get_fixed_length_mouse_movement('user016', 'session_2020_05_25_1min.csv')
    #fit_and_evaluate_network('user001')
    #data, labels = get_identification_dataset()
    #plot_datasets_distributions()
    #test()
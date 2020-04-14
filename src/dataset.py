import os
import random

import pandas as pd
import numpy as np
from glob import glob
from math import inf, ceil
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# User defined imports
import config.settings as stt
import config.constants as const

# 0 - valid user (is legal)
POSITIVE_CLASS = 0
# 1 - intruder (is illegal)
NEGATIVE_CLASS = 1

class Dataset:
    __instance = None


    @staticmethod 
    def getInstance():
        """ Static access method. 
        """
        if Dataset.__instance == None:
            Dataset()
        return Dataset.__instance


    def __init__(self):
        """ Virtually private constructor. 
        """
        if Dataset.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Dataset.__instance = self


    if const.VERBOSE:
        def print_msg(self, msg):
            """ Prints the given message if VERBOSE is True

                Parameters: msg (str)

                Returns:
                    void
            """
            print(msg)
    else:
        print_msg = lambda msg: None


    def __load_session(self, session_path, n_rows = None):
        """ Reads the given columns from .csv file

            Parameters: session_path (str): the full path of the given session

            Returns:
                DataFrame: returning value
        """
        try:

            if n_rows is not None:
                return pd.read_csv(session_path, usecols=['x', 'y', 'client timestamp', 'button'], nrows=n_rows)[['x', 'y', 'client timestamp', 'button']]
            else:
                return pd.read_csv(session_path, usecols=['x', 'y', 'client timestamp', 'button'])[['x', 'y', 'client timestamp', 'button']]

        except BaseException:
            raise Exception("Can't open file " + session_path)


    def __load_user_sessions(self, user, block_num, files_path):
        """ Loads a given user all sessions
            
            Parameters: 
                user (str): user name
                files_path (str): specifies the input file location

            Returns:
                DataFrame: returning value
        """
        # Maximum number of rows to read
        # We multiply the value by 3, because of the unknown chunk samples size
        # It will be an overestimation of the useful
        if block_num == inf:
            n_rows = None
        else:
            n_rows = 3 * const.BLOCK_SIZE * block_num

        sessions_data_df = pd.DataFrame()
        # Iterate through all user sessions
        for session_path in glob(files_path + '/' + user + '/*'):
            sessions_data_df = pd.concat([sessions_data_df, self.__load_session(session_path, n_rows)])

            # If we collected enough data than we stop reading further sessions
            if n_rows is not None and sessions_data_df.shape[0] > n_rows:
                break

        if sessions_data_df.empty:
            raise Exception("Can't iterate through files: " + files_path + '/' + user)

        return sessions_data_df


    def __filter_by_states(self, df):
        """ Filters the given states from dataset

            Parameters:
                df (DataFrame): input data

            Returns:
                DataFrame: returning value
        """
        # Filtering outliers
        df = df[(df['x'] < const.MAX_WIDTH) & (df['y'] < const.MAX_HEIGHT)]
        # Dropping the Scroll states
        return df.loc[~df['button'].isin(['Scroll'])]


    def __get_handled_raw_data(self, df, block_num):
        """ Handles the idle state with method that is defined by sel_chunck_samples_handler

            Parameters:
                df (DataFrame): input data
                block_num (int): number of rows to return

            Return:
                DataFrame: preprocessed dx, dy and dt with chunck handled
        """
        complete_df, chunk_df = self.__handle_raw_data(df)

        if stt.sel_chunck_samples_handler == stt.ChunkSamplesHandler.CONCATENATE_CHUNKS:
            complete_df = pd.concat([complete_df, chunk_df], axis=0)

        row_num = const.BLOCK_SIZE * block_num

        if row_num < complete_df.shape[0]:
            return complete_df[:row_num]

        return complete_df


    def __handle_raw_data(self, df):
        """ Separates the data (df) to complete_df and chunk_df.
            complete_df holds samples, that is equal or greater than BLOCK_NUM
            chunk_df holds samples, that is less than BLOCK_NUM
            Parameters:
                df (DataFrame): input dataframe, that contains all user session
            Returns:
                DataFrame: df1 (DataFrame), df2 (DataFrame)
                    df1 contains the entire samples that with in given BLOCK_SIZE
                    df2 contains the chunk, which length is less than the BLOCK_SIZE
        """
        # Calculates the difference between two consecutive row
        df = df.diff()

        # The first row values are NaN, because of using diff() 
        df = df[1:].rename(columns={'x': 'dx', 'y': 'dy', 'client timestamp': 'dt'})

        default_time = 0.016
        # Setting default value if dt == 0
        df.loc[ df['dt'] <= 0.01, 'dt' ] = default_time

        df = df.reset_index(drop=True)
        outlays_index_list = np.concatenate(([0], df.loc[ df['dt'] > const.STATELESS_TIME ].index), axis=0)

        # Resetting the outlayer values
        df.loc[ df['dt'] > const.STATELESS_TIME, 'dt' ] = default_time

        chunk_samples_indexes = []
        for i in range(1, len(outlays_index_list)):
            reminder = (outlays_index_list[i] - outlays_index_list[i - 1]) % const.BLOCK_SIZE
            quotient = (outlays_index_list[i] - outlays_index_list[i - 1]) / const.BLOCK_SIZE

            if (reminder != 0) or quotient.is_integer():
                chunk_samples_indexes.extend(range(outlays_index_list[i] - reminder, outlays_index_list[i]))

        #for i in range(len(chunk_samples_indexes)):
        #    chunk_samples_indexes[i] = chunk_samples_indexes[i] - 1

        # Return complete_samples and chunk_samples
        return df.loc[~df.index.isin(chunk_samples_indexes)], df.iloc[chunk_samples_indexes]


    def __get_velocity_from_data(self, df):
        """ Returns velocity from raw data

            Parameters:
                df (DataFrame): input dataframe, it contains dx, dy and dt

            Returns:
                DataFrame: x and y directional speed
        """

        df['vx'] = df['dx'] / df['dt']
        df['vy'] = df['dy'] / df['dt']

        return pd.concat([df['vx'], df['vy']], axis=1)


    def __get_shift_from_data(self, df):
        """ Returns velocity from raw data

            Parameters:
                df (DataFrame): input dataframe, it contains dx, dy and dt

            Returns:
                DataFrame: horizontal and vertical shift components
        """

        return pd.concat([df['dx'], df['dy']], axis=1)


    def __get_shaped_dataset_by_user(self, user, block_num, files_path):
        """ Returns data with shape of (BLOCK_NUM, BLOCK_SIZE, 2)

            Parameters:
                user (np.ndarray): selected user
                block_num (int): number of rows to return
                files_path (str): specifies the input file location

            Returns:
                np.ndarray: given shape of ndarray
        """
        data = self.__get_preprocessed_data(user, block_num, files_path)

        return np.reshape(data, (int(data.shape[0] / const.BLOCK_SIZE), const.BLOCK_SIZE, 2))


    def __get_preprocessed_data(self, user, block_num, files_path):
        """ Returns a given user preprocessed sessions data

            Parameters:
                user (str): selected user name
                block_num (int): number of rows to return
                files_path (str): specifies the input file location

            Returns:
                np.ndarray: given shape of ndarray containing v_x and v_y
        """
        # If the user is None it means that we only need to read one given session, specified with files_path parameter
        if user == None:
            data = self.__filter_by_states( self.__load_session(files_path, inf) )
        else:
            data = self.__filter_by_states( self.__load_user_sessions(user, block_num, files_path) )

        data = self.__get_handled_raw_data(data[['x', 'y', 'client timestamp']], block_num)

        # Checking if we have enough data
        if data.shape[0] < const.BLOCK_SIZE * block_num and block_num != inf:
            print('Data augmented for user: ', user, '####################################')
            data = self.__get_augmentated_dataset(data, block_num)

        data = self.__get_features_from_raw_data(data)

        # Slicing array to fit in the given shape
        row_num_end = int(data.shape[0] / const.BLOCK_SIZE) * const.BLOCK_SIZE

        return data[:row_num_end].values


    def __get_features_from_raw_data(self, df):

        if stt.sel_raw_feature_type == stt.RawFeatureType.VX_VY:
            data = self.__get_velocity_from_data( df )

        if stt.sel_raw_feature_type == stt.RawFeatureType.DX_DY:
            data = self.__get_shift_from_data( df )

        return data


    def __get_augmentated_dataset(self, df, block_num):
        """ Augments the data for the given size, set in constants.py

            Parameters:
                df (DataFrame): input data
                block_num (int): number of rows to return

            Returns:
                df (DataFrame): augmentated dataset
        """
        # Calculating how many rows have to augment
        if block_num == inf:
            aug_row_num = df.shape[0]
        else:
            aug_row_num = const.BLOCK_SIZE * block_num - df.shape[0]
        
        aug_df = df[:aug_row_num].copy()

        # Concatenating given values until it reaches the given size
        while aug_df.shape[0] < aug_row_num:
            aug_df = pd.concat([aug_df, aug_df])

        # Checking if concatenated df is bigger than the expected
        if aug_df.shape[0] > aug_row_num:
            aug_df = aug_df[:aug_row_num]

        # Applying an augmentation function to all columns data
        aug_df.loc[:] -= 0.1 + 0.2 * (random.random() * 2 - 1)

        return pd.concat([df, aug_df], axis=0)
        

    def __load_positive_dataset(self, user, block_num, files_path):
        """ Loads a given user all sessions

            Parameters:
                user (str): selected user name
                block_num (int): number of rows to return 
                file_path (str): given files path

            Returns:
                np.ndarray: given shape of ndarray
        """
        return self.__get_shaped_dataset_by_user(user, block_num, files_path)


    def __load_negative_dataset(self, user, block_num, files_path):
        """ Loads a dataset from all other users with a given size

            Parameters:
                user (str): selected user name
                block_num (int): number of rows to read from each user
                files_path (str): defines the path of the given files

            Returns:
                np.ndarray: given shape of ndarray
        """
        users = os.listdir(files_path)
        # Remove the current user from the list
        users.remove(user)
        
        dataset = np.ndarray(shape=(0, const.BLOCK_SIZE, 2), dtype=float)
        for user in users:
            dataset = np.concatenate((dataset, self.__load_positive_dataset(user, block_num, files_path)), axis=0)

        return dataset


    def __load_positive_balanced_dataset(self, user, block_num, files_path):
        """ Loads positive balanced dataset.
            Reads block_num positive dataset and block_num / #users negative dataset

            Parameters:
                user (str): selected user name
                block_num (int): number of rows containing each dataset
                files_path (str): defines the path of the given files

            Returns:
                np.ndarray: positive and negative data concatenation
        """
        dset_positive = self.__load_positive_dataset(user, stt.BLOCK_NUM, files_path)

        # Calculates how many block_num has to read from users
        user_num = len(os.listdir(files_path)) - 1
        # We take greater or equal block_num from users than the current user's block_num
        negative_block_num = ceil(dset_positive.shape[0] / user_num)
        dset_negative = self.__load_negative_dataset(user, negative_block_num, files_path)

        # Checking if we selected more data then the required
        if dset_positive.shape[0] < dset_negative.shape[0]:
            dset_negative = dset_negative[:dset_positive.shape[0]]

        return np.concatenate((dset_positive, dset_negative), axis=0)


    def __load_negative_balanced_dataset(self, user, block_num, files_path):
        """ Loads positive balanced dataset.
            Reads block_num negative dataset and block_num / #users positive dataset

            Parameters:
                user (str): selected user name
                block_num (int): number of rows containing each dataset
                files_path (str): defines the path of the given files

            Returns:
                np.ndarray: positive and negative data concatenation
        """
        dset_negative = self.__load_negative_dataset(user, block_num, files_path)
        dset_positive = self.__load_positive_dataset(user, dset_negative.shape[0], files_path)

        return np.concatenate((dset_positive, dset_negative), axis=0)


    def __create_labeled_dataset(self, user):
        """ Creates dataset with labels

            Parameters: 
                user (str) - username

            Returns:
                np.ndarray, np.ndarray: dataset, labels
        """
        if stt.sel_balance_type == stt.DatasetBalanceType.POSITIVE:
            data = self.__load_positive_balanced_dataset(user, stt.BLOCK_NUM, const.TRAIN_FILES_PATH)
        else:
            data = self.__load_negative_balanced_dataset(user, stt.BLOCK_NUM, const.TRAIN_FILES_PATH)

        # 0 - valid user (is legal)
        # 1 - intruder (is illegal)
        labels = np.concatenate((self.__create_labels(int(data.shape[0] / 2), POSITIVE_CLASS), self.__create_labels(int(data.shape[0] / 2), NEGATIVE_CLASS)))

        return data, labels


    def __create_labels(self, length, value):
        """ Creates an array full of the given number (value)

            Parameters:
                length (int): array length
                value (int): fills the array with this value

            Returns:
                np.ndarray: given length of array filled with number defined by value
        """
        return np.full((length, 1), value)


    def create_train_dataset_for_authentication(self, user):

        if stt.sel_authentication_type == stt.AuthenticationType.BINARY_CLASSIFICATION:
            return self.__create_train_dataset_authentication_for_binary_classification(user)

        if stt.sel_authentication_type == stt.AuthenticationType.ONE_CLASS_CLASSIFICATION:
            return self.__create_train_dataset_authentication_for_one_class_classification(user)


    def __create_train_dataset_authentication_for_one_class_classification(self, user):
        pass


    def __create_train_dataset_authentication_for_binary_classification(self, user):
        """ Returns the train dataset and the labels for the dataset

            Parameters:
                user (str) - username

            Returns:
                np.ndarray, np.array: train dataset, train labels for the dataset
        """
        data, labels = self.__create_labeled_dataset(user)
        print(data.shape)

        # If we only have train dataset we split into train and test data
        if stt.sel_dataset_type == stt.DatasetType.TRAIN_AVAILABLE:
            data, labels = self.__split_and_scale_dateset(data, labels)

        return data, labels


    def __create_test_dataset_authentication_for_one_class_classification(self, user):
        pass


    def create_test_dataset_for_authentication(self, user):
        """ Returns the test dataset and the labels for the dataset

            Parameters:
                user (str) - username

            Returns:
                np.ndarray, np.array: test dataset, test labels for the dataset
        """
        if stt.sel_authentication_type == stt.AuthenticationType.BINARY_CLASSIFICATION:

            if stt.sel_dataset_type == stt.DatasetType.TRAIN_TEST_AVAILABLE:
                return self.__get_train_test_available_test_dataset(user)
            if stt.sel_dataset_type == stt.DatasetType.TRAIN_AVAILABLE:
                return self.__get_train_available_test_dataset(user)

        if stt.sel_authentication_type == stt.AuthenticationType.ONE_CLASS_CLASSIFICATION:
            return self.__create_test_dataset_authentication_for_one_class_classification(user)


    def __split_and_scale_dateset(self, data, labels):

        if stt.sel_user_recognition_type == stt.UserRecognitionType.AUTHENTICATION:
            X_train, X_test, y_train, y_test = self.__split_data_authentication(data, labels)
        
        if stt.sel_user_recognition_type == stt.UserRecognitionType.IDENTIFICATION:
            X_train, X_test, y_train, y_test = self.__split_data_identification(data, labels)

        X_train, X_test = self.__scale_dataset(X_train, X_test)

        if stt.sel_method == stt.Method.TRAIN:
            return X_train, y_train

        if stt.sel_method == stt.Method.EVALUATE:
            return X_test, y_test


    def __scale_dataset(self, X_train, X_test):
        return self.__scale_data_with_selected_scaler(stt.sel_scaling_method.value, X_train, X_test)


    def __scale_data_with_selected_scaler(self, arg, X_train, X_test):
        """ Scale data with specified method

            Parameters:
                df (DataFrame): input dataframe, that contains al user session

            Returns:
        """
        switcher = { 
            0: self.__user_defined_scaler,
            1: self.__min_max_scaler,
            2: self.__max_abs_scaler,
            3: self.__no_scaler,
            4: self.__standard_scaler
        } 

        func = switcher.get(arg, lambda: "Wrong scaler!")
        return func(X_train, X_test)


    def __user_defined_scaler(self, X_train, X_test):
        """ Makes max elem scaling along y axis

            Parameters:
                df (DataFrame): input dataframe

            Returns:
                DataFrame: scaled dataframe
        """
        max_x = np.max( np.absolute(X_train[:, :, 0]) )
        max_y = np.max( np.absolute(X_train[:, :, 1]) )
        
        X_train[:, :, 0] = X_train[:, :, 0] / max_x
        X_train[:, :, 1] = X_train[:, :, 1] / max_y
        X_test[:, :, 0] = X_test[:, :, 0] / max_x
        X_test[:, :, 1] = X_test[:, :, 1] / max_y

        return X_train, X_test


    def __min_max_scaler(self, X_train, X_test):

        min_max_scaler_x = preprocessing.MinMaxScaler(feature_range=(0, 1))
        min_max_scaler_y = preprocessing.MinMaxScaler(feature_range=(0, 1))

        min_max_scaler_x.fit(X_train[:, :, 0])
        X_train[:, :, 0] = min_max_scaler_x.transform(X_train[:, :, 0])
        min_max_scaler_y.fit(X_train[:, :, 1])
        X_train[:, :, 1] = min_max_scaler_y.transform(X_train[:, :, 1])

        X_test[:, :, 0] = min_max_scaler_x.transform(X_test[:, :, 0])
        X_test[:, :, 1] = min_max_scaler_y.transform(X_test[:, :, 1])

        return X_train, X_test


    def __max_abs_scaler(self, X_train, X_test):

        max_abs_scaler_x = preprocessing.MaxAbsScaler()
        max_abs_scaler_y = preprocessing.MaxAbsScaler()

        max_abs_scaler_x.fit(X_train[:, :, 0])
        X_train[:, :, 0] = max_abs_scaler_x.transform(X_train[:, :, 0])
        max_abs_scaler_y.fit(X_train[:, :, 1])
        X_train[:, :, 1] = max_abs_scaler_y.transform(X_train[:, :, 1])

        X_test[:, :, 0] = max_abs_scaler_x.transform(X_test[:, :, 0])
        X_test[:, :, 1] = max_abs_scaler_y.transform(X_test[:, :, 1])

        return X_train, X_test

    
    def __no_scaler(self, X_train, X_test):
        return X_train, X_test


    def __standard_scaler(self, X_train, X_test):

        mean_x = np.mean(X_train[:, :, 0], axis=0)
        std_x = np.std(X_train[:, :, 0], axis=0)
        mean_y = np.mean(X_train[:, :, 1], axis=0)
        std_y = np.std(X_train[:, :, 1], axis=0)

        X_train[:, :, 0] = (X_train[:, :, 0] - mean_x) / std_x
        X_train[:, :, 1] = (X_train[:, :, 1] - mean_y) / std_y
        X_test[:, :, 0] = (X_test[:, :, 0] - mean_x) / std_x
        X_test[:, :, 1] = (X_test[:, :, 1] - mean_y) / std_y

        return X_train, X_test


    def __split_data_authentication(self, data, labels):
        """ Splits data and labels with a predifined ratio

            Parameters:
                data (np.ndarray): dataset
                labels (np.ndarray): labels for the given dataset

            Returns:
                np.ndarray, np.array: splitted train dataset, splitted train labels for the dataset
        """
        # Getting the barrier position between positive and negative dataset
        # data[middel_pos:, :, :] contains the positive dataset
        # data[:middel_pos, :, :] contains the negative dataset
        middle_pos = int(data.shape[0] / 2)

        # Inverting train_test_split() to get test data from the beginning of the dataset
        if const.TRAIN_TEST_SPLIT_VALUE <= 1:
            test_size = 1 - int(const.TRAIN_TEST_SPLIT_VALUE / 2)
        else:
            test_size = middle_pos - int(const.TRAIN_TEST_SPLIT_VALUE / 2)
        # We divide by 2, because of the positive and negative samples

        X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(data[middle_pos:], labels[middle_pos:], test_size=test_size, random_state=const.RANDOM_STATE, shuffle=False)
        X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(data[:middle_pos], labels[:middle_pos], test_size=test_size, random_state=const.RANDOM_STATE, shuffle=False)

        X_train = np.concatenate((X_test_pos, X_test_neg), axis=0)
        X_test = np.concatenate((X_train_pos, X_train_neg), axis=0)
        y_train = np.concatenate((y_test_pos, y_test_neg), axis=0)
        y_test = np.concatenate((y_train_pos, y_train_neg), axis=0)

        return X_train, X_test, y_train, y_test


    def __split_data_identification(self, data, labels):
        """ Splits data and labels with a predifined ratio
            Dataset contains only positive samples
            Used for creating train dataset

            Parameters:
                data (np.ndarray): dataset
                labels (np.ndarray): labels for the given dataset

            Returns:
                np.ndarray, np.array: splitted train dataset, splitted train labels for the dataset
        """
        if const.TRAIN_TEST_SPLIT_VALUE <= 1:
            test_size = 1 - const.TRAIN_TEST_SPLIT_VALUE
        else:
            test_size = data.shape[0] - const.TRAIN_TEST_SPLIT_VALUE

        X_test, X_train, y_test, y_train = train_test_split(data, labels, test_size=test_size, random_state=const.RANDOM_STATE, shuffle=False)

        return X_train, X_test, y_train, y_test


    def __get_train_available_test_dataset(self, user):
        """ Returns the dataset splitted to test dataset and the labels for it

            Parameters:
                user (str) - username

            Returns:
                np.ndarray, np.array: test dataset, test labels for the dataset
        """
        data, labels = self.__create_labeled_dataset(user)
        
        return self.__split_and_scale_dateset(data, labels)

    
    def __get_train_test_available_test_dataset(self, user):
        """ Returns the dataset from test files, which is labeled in public labels file

            Parameters:
                user (str) - username

            Returns:
                np.ndarray, np.array: test dataset, test labels for the dataset
        """
        # Contains the public labels (session name and given session labeled value)
        labels_df = pd.read_csv(const.TEST_LABELS_PATH, names=['session', 'label'])
        dataset = np.ndarray(shape=(0, const.BLOCK_SIZE, 2), dtype=float)
        labels = np.ndarray(shape=(0, 1), dtype=int)

        for session_path in glob(const.TEST_FILES_PATH + '/' + user + '/*'):
            session_name = session_path[len(session_path) - 18 : ]

            # If the current session is labeled in public labels file
            if not labels_df.loc[labels_df['session'] == session_name].empty:
                # Getting the session labeled value (int)
                label = labels_df.loc[labels_df['session'] == session_name, 'label'].index.item()
                # None - for reading only one session file, specified with session_path
                # inf - because it reads only the original data, not cropped or augmented
                # session_path - the given session path to read
                tmp_dset = self.__load_positive_dataset(None, inf, session_path)
                dataset = np.concatenate((dataset, tmp_dset), axis=0)
                labels = np.concatenate((labels, self.__create_labels(tmp_dset.shape[0], label)), axis=0)

        return dataset, labels


    def __create_identification_dataset(self):
        """ Returns the identification dataset defined with method

            Parameters:
                method (str) - Value in (TRAIN, EVALUATE, TRANFER LEARNING)

            Returns:
                np.ndarray, np.array: train/test dataset, train/test labels for identification
        """
        # Listing all user in a given dataset
        users = os.listdir(const.TRAIN_FILES_PATH)
        dataset = np.ndarray(shape=(0, const.BLOCK_SIZE, 2))
        labels = np.ndarray(shape=(0, 1))

        # We have to renumber user ids, because of using to_categorical()
        id = 0
        for user in users:
            tmp_dataset = self.__load_positive_dataset(user, stt.BLOCK_NUM, const.TRAIN_FILES_PATH)
            tmp_labels = self.__create_labels(tmp_dataset.shape[0], id)

            tmp_dataset, tmp_labels = self.__split_and_scale_dateset(tmp_dataset, tmp_labels)

            id = id + 1
            self.print_msg('Dataset shape from user: ' +  user + ' - ' + str(tmp_dataset.shape))

            dataset = np.concatenate((dataset, tmp_dataset), axis=0)
            labels = np.concatenate((labels, tmp_labels), axis=0)
    
        return dataset, labels


    def create_train_dataset_for_identification(self):
        """ Returns identification train dataset

            Parameters:
                None

            Returns:
                np.ndarray, np.array: train dataset, train labels for the identification
        """
        return self.__create_identification_dataset()


    def create_test_dataset_for_identification(self):
        """ Returns identification test dataset

            Parameters:
                None

            Returns:
                np.ndarray, np.array: test dataset, test labels for the identification
        """
        return self.__create_identification_dataset()
               

    # Statistics ----------------------------------------------------------------------------------------------


    def print_all_user_dataset_shape(self):
        """ Prints available dataset shape for all user

            Parameters:
                None

            Returns:
                None
        """
        users = os.listdir(const.TRAIN_FILES_PATH)

        for user in users:
            dataset = self.__load_positive_dataset(user, inf, const.TRAIN_FILES_PATH)
            print('Dataset shape for user:', user , '-', dataset.shape)
            
            
    def __get_raw_user_data(self, user, block_num, files_path):

        data = self.__filter_by_states( self.__load_user_sessions(user, block_num, files_path) )
        data = self.__get_handled_raw_data(data[['x', 'y', 'client timestamp']], block_num)

        # Checking if we have enough data
        if data.shape[0] < const.BLOCK_SIZE * block_num and block_num != inf:
            print('Data augmented for user: ', user, '####################################')
            data = self.__get_augmentated_dataset(data, block_num)


        # Slicing array to fit in the given shape
        row_num_end = int(data.shape[0] / const.BLOCK_SIZE) * const.BLOCK_SIZE

        return data[:row_num_end]


    def get_raw_identification_data(self):
        return self.__get_raw_user_data('user9', stt.BLOCK_NUM, const.TRAIN_FILES_PATH)


            
if __name__ == "__main__":
    dataset = Dataset()
    #x_train, y_train = dataset.create_train_dataset_for_identification()
    #dataset.print_all_user_dataset_shape()
    #x_train = dataset.create_train_dataset_for_authentication(const.USER_NAME)
    x_train, y_train = dataset.create_train_dataset_for_authentication(const.USER_NAME)
    #print(x_train[0], ' x_test shape')
    #print(y_test.shape, ' y_tets shape')
    print(type(x_train))
    print(x_train.shape)

    #dataset.print_preprocessed_identification_dataset()
    #dataset.print_preprocessed_identification_dataset_vx_vy_separate()
    #trainX, trainy = dataset.create_train_dataset_for_identification()
    #print(trainX.shape, ' trainX shape')
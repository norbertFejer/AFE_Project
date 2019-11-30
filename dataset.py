import pandas as pd
import numpy as np
from glob import glob
import os
import random
from math import inf, ceil
from sklearn.model_selection import train_test_split

# User defined imports
import constants as const
import settings as stt

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
            """ Prints the given message

                Parameters: msg (str)

                Returns:
                    void
            """
            print(msg)
    else:
        print_msg = lambda msg: None


    def __load_session(self, session_path):
        """ Reads the given columns from .csv file

            Parameters: session_path (str): the full path of the given session

            Returns:
                DataFrame: returning value
        """
        try:
            return pd.read_csv(session_path, usecols=['x', 'y', 'client timestamp', 'button'])[['x', 'y', 'client timestamp', 'button']]
        except BaseException:
            raise Exception("Can't open file " + session_path)


    def __load_user_sessions(self, user, files_path):
        """ Loads a given user all sessions
            
            Parameters: 
                user (str): user name
                files_path (str): specifies the input file location

            Returns:
                DataFrame: returning value
        """
        sessions_data_df = pd.DataFrame()
        # Iterate through all user sessions
        for session_path in glob(files_path + '/' + user + '/*'):
            sessions_data_df = pd.concat([sessions_data_df, self.__load_session(session_path)])

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
            ret_data = pd.concat([complete_df, chunk_df], axis=0)
        else:
            ret_data = complete_df

        row_num = const.BLOCK_SIZE * block_num

        if row_num < ret_data.shape[0]:
            return ret_data[:row_num]

        return ret_data


    def __handle_raw_data(self, df):
        """ Separates the data (df) to compelete_df and chunk_df.
            complete_df holds samples, that is equal or greater than BLOCK_NUM
            chunk_df holds samples, that is less than BLOCK_NUM

            Parameters:
                df (DataFrame): input dataframe, that contains al user session

            Returns:
                DataFrame: df1 (DataFrame), df2 (DataFrame)
                    df1 contains the entire samples that with in given BLOCK_SIZE
                    df2 contains the chunkc, which length is less than the BLOCK_SIZE
        """

        # Calculates the difference between two consecutive row
        df = df.diff()
        df = df.rename(columns={'x': 'dx', 'y': 'dy', 'client timestamp': 'dt'})

        # Gets the index values, where dt is greater than STATELESS_TIME
        outlays_index_list = np.where((df['dt'] > const.STATELESS_TIME) & (df['dt'] > 0))[0]
        # -1 value is a pivot for using later in the code
        outlays_index_list = np.append([outlays_index_list], [-1])

        # Vectorization for better performance
        df = df.values
        # complete_samples are samples that are greater or equal than the BLOCK_SIZE
        # chunk_samples are lengths less than BLOCK_SIZE
        complete_samples, chunk_samples = [], []
        row_start, row_end, ind = 1, 1 + const.BLOCK_SIZE, 0

        while row_end < df.shape[0]:

            row_end = row_start + const.BLOCK_SIZE

            # Checking if sample contains outlays
            # or all outlays in the list processed yet
            if row_end < outlays_index_list[ind] or outlays_index_list[ind] == -1:
                complete_samples.append(df[row_start:row_end])
                row_start = row_end
            else:
                chunk_samples.append(df[row_start:outlays_index_list[ind]])
                # Next sample starts at the position of the actual outlay
                row_start = outlays_index_list[ind]
                ind += 1

        # Checking if we have another chunks left
        if row_end < df.shape[0]:
            chunk_samples.append(df[row_end:df.shape[0]])

        complete_samples = np.concatenate(complete_samples, axis=0) if len(complete_samples) != 0 else np.ndarray(shape=(0, 3))
        chunk_samples = np.concatenate(chunk_samples, axis=0) if len(chunk_samples) != 0 else np.ndarray(shape=(0, 3))

        # Return values as df with given column names
        return pd.DataFrame(complete_samples, columns=['dx', 'dy', 'dt']), pd.DataFrame(chunk_samples, columns=['dx', 'dy', 'dt'])


    def __normalize_data(self, df):
        """ Normalize data with specified method

            Parameters:
                df (DataFrame): input dataframe, that contains al user session

            Returns:
                DataFrame: normalized dataframe
        """

        if stt.sel_normalization_method == stt.NormalizationMethod.USER_DEFINED:
            self.__normalize_data_user_defined(df)
        else:
            return 2


    def __normalize_data_user_defined(self, df):
        """ Makes max elem normalization along y axis

            Parameters:
                df (DataFrame): input dataframe

            Returns:
                DataFrame: normalized dataframe
        """

        df['v_x'] /= abs(df['v_x']).max()
        df['v_y'] /= abs(df['v_y']).max()


    def __normalize_data_builtin(self, df):
        # TODO
        return 1


    def __get_velocity_from_data(self, df):
        """ Returns velocity from raw data

            Parameters:
                df (DataFrame): input dataframe, it contains dx, dy and dt

            Returns:
                DataFrame: x and y directional speed
        """

        # Setting default value if dt == 0
        df.loc[ df['dt'] <= 0 ] = 0.01

        df['v_x'] = df['dx'] / df['dt']
        df['v_y'] = df['dy'] / df['dt']

        return pd.concat([df['v_x'], df['v_y']], axis=1)


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
            data = self.__filter_by_states( self.__load_session(files_path) )
        else:
            data = self.__filter_by_states( self.__load_user_sessions(user, files_path) )

        data = self.__get_velocity_from_data( self.__get_handled_raw_data(data[['x', 'y', 'client timestamp']], block_num) )
        self.__normalize_data(data)

        # Checking if we have enough data
        if data.shape[0] < const.BLOCK_SIZE * block_num and block_num != inf:
            data = self.__get_augmentated_dataset(data, block_num)

        # Slicing array to fit in the given shape
        row_num_end = int(data.shape[0] / const.BLOCK_SIZE) * const.BLOCK_SIZE

        return data[:row_num_end].values


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
        dset_positive = self.__load_positive_dataset(user, const.BLOCK_NUM, files_path)

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
            data = self.__load_positive_balanced_dataset(user, const.BLOCK_NUM, const.TRAIN_FILES_PATH)
        else:
            data = self.__load_negative_balanced_dataset(user, const.BLOCK_NUM, const.TRAIN_FILES_PATH)

        # 0 - valid user (is legal)
        # 1 - intruder (is illegal)
        labels = np.concatenate((self.__create_labels(int(data.shape[0] / 2), 0), self.__create_labels(int(data.shape[0] / 2), 1)))

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


    def create_train_dataset(self, user):
        """ Returns the train dataset and the labels for the dataset

            Parameters:
                user (str) - username

            Returns:
                np.ndarray, np.array: train dataset, train labels for the dataset
        """
        data, labels = self.__create_labeled_dataset(user)

        # If we only have train dataset we split into train and test data
        if stt.sel_dataset_type == stt.DatasetType.TRAIN_AVAILABLE:
            data, labels = self.__split_dataset_to_train(data, labels)

        return data, labels


    def create_test_dataset(self, user):
        """ Returns the test dataset and the labels for the dataset

            Parameters:
                user (str) - username

            Returns:
                np.ndarray, np.array: test dataset, test labels for the dataset
        """
        if stt.sel_dataset_type == stt.DatasetType.TRAIN_TEST_AVAILABLE:
            return self.__get_train_test_available_test_dataset(user)
        else:
            return self.__get_train_available_test_dataset(user)
        


    def __split_dataset_to_train(self, data, labels):
        """ Splits data and labels with a predifined ratio

            Parameters:
                data (np.ndarray): dataset
                labels (np.ndarray): labels for the given dataset

            Returns:
                np.ndarray, np.array: splitted train dataset, splitted train labels for the dataset
        """
        # Getting the barrier position between positive and negative dataset
        # data[:middel_pos] contains the positive dataset
        # data[middel_pos:] contains the negative dataset
        middle_pos = int(data.shape[0] / 2)
        # Splitting the positive dataset
        pos_data, _, pos_labels, _ = train_test_split(data[:middle_pos], labels[:middle_pos], test_size=const.TRAIN_TEST_SPLIT_VALUE, random_state=const.RANDOM_STATE, shuffle=False)
        
        return np.concatenate((pos_data, data[middle_pos : (middle_pos + pos_data.shape[0])]), axis=0), \
               np.concatenate((pos_labels, labels[middle_pos: (middle_pos + pos_data.shape[0])]), axis=0)


    def __split_dataset_to_test(self, data, labels):
        """ Splits data and labels with a predifined ratio
            Dataset contains both positive and negative samples

            Parameters:
                data (np.ndarray): dataset
                labels (np.ndarray): labels for the given dataset

            Returns:
                np.ndarray, np.array: splitted test dataset, splitted test labels for the dataset
        """
        middle_pos = int(data.shape[0] / 2)
        _, pos_data, _, pos_labels = train_test_split(data[:middle_pos], labels[:middle_pos], test_size=const.TRAIN_TEST_SPLIT_VALUE, random_state=const.RANDOM_STATE, shuffle=False)

        return np.concatenate((pos_data, data[(data.shape[0] - pos_data.shape[0]) : ]), axis=0), \
               np.concatenate((pos_labels, labels[(data.shape[0] - pos_data.shape[0]) : ]), axis=0)


    def __get_train_available_test_dataset(self, user):
        """ Returns the dataset splitted to test dataset and the labels for it

            Parameters:
                user (str) - username

            Returns:
                np.ndarray, np.array: test dataset, test labels for the dataset
        """
        data, labels = self.__create_labeled_dataset(user)

        return self.__split_dataset_to_test(data, labels)

    
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


    def __create_identification_dataset_by_method(self, method):
        users = os.listdir(const.TRAIN_FILES_PATH)
        dataset = np.ndarray(shape=(0, const.BLOCK_SIZE, 2))
        labels = np.ndarray(shape=(0, 1))

        id = 0
        for user in users:
            tmp_dataset = self.__load_positive_dataset(user, const.BLOCK_NUM, const.TRAIN_FILES_PATH)
            tmp_labels = self.__create_labels(tmp_dataset.shape[0], id)

            if method == stt.Method.TRAIN:
                tmp_dataset, tmp_labels = self.__split_positive_data_to_train_dataset(tmp_dataset, tmp_labels)
            else:
                tmp_dataset, tmp_labels = self.__split_positive_data_to_test_dataset(tmp_dataset, tmp_labels)

            id = id + 1
            self.print_msg('Dataset shape from user: ' +  user + ' - ' + str(tmp_dataset.shape))

            dataset = np.concatenate((dataset, tmp_dataset), axis=0)
            labels = np.concatenate((labels, tmp_labels), axis=0)
    
        return dataset, labels


    def create_train_dataset_for_identification(self):
        return self.__create_identification_dataset_by_method(stt.Method.TRAIN)


    def create_test_dataset_for_identification(self):
        return self.__create_identification_dataset_by_method(stt.Method.EVALUATE)


    def __split_positive_data_to_train_dataset(self, data, labels):
        """ Splits data and labels with a predifined ratio
            Dataset contains only positive samples

            Parameters:
                data (np.ndarray): dataset
                labels (np.ndarray): labels for the given dataset

            Returns:
                np.ndarray, np.array: splitted train dataset, splitted train labels for the dataset
        """
        # Splitting the positive dataset
        trainX, _, trainy, _ = train_test_split(data, labels, test_size=const.TRAIN_TEST_SPLIT_VALUE, random_state=const.RANDOM_STATE, shuffle=False)
        
        return trainX, trainy


    def __split_positive_data_to_test_dataset(self, data, labels):
        """ Splits data and labels with a predifined ratio
            Dataset contains only positive samples

            Parameters:
                data (np.ndarray): dataset
                labels (np.ndarray): labels for the given dataset

            Returns:
                np.ndarray, np.array: splitted train dataset, splitted train labels for the dataset
        """
        # Splitting the positive dataset
        _, testX, _, testy = train_test_split(data, labels, test_size=const.TRAIN_TEST_SPLIT_VALUE, random_state=const.RANDOM_STATE, shuffle=False)
        
        return testX, testy
               

    # Statistics ----------------------------------------------------------------------------------------------

    def print_all_user_dataset_shape(self):
        users = os.listdir(const.TRAIN_FILES_PATH)

        for user in users:
            dataset = self.__load_positive_dataset(user, inf, const.TRAIN_FILES_PATH)
            print('Dataset shape for user:', user , '-', dataset.shape)


    # Plotter ----------------------------------------------------------------------------------------------

    def get_user_preprocessed_dataset(self, user): 
        data = self.__filter_by_states( self.__load_user_sessions(user, const.TRAIN_FILES_PATH) )
        data = self.__get_handled_raw_data(data[['x', 'y', 'client timestamp']], inf)
    
        return data
            
            
if __name__ == "__main__":
    dataset = Dataset()
    dataset.create_test_dataset_for_identification()
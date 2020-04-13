from enum import Enum
import os

import numpy as np

import src.dataset as dset
import config.constants as const
import config.settings as stt

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

class SaveValues(Enum):
    VX = 1
    VY = 2
    VX_VY = 3


class GenerateCSV:


    def __init__(self):
        self.__dataset = dset.Dataset()
        self.__saved_values_folder_name = "generatedCSVData"

        if stt.sel_user_recognition_type == stt.UserRecognitionType.IDENTIFICATION:
            self.__get_dataset_for_identicitaion()

        self.__create_users_list()

        if not os.path.exists(self.__saved_values_folder_name):
            os.makedirs(self.__saved_values_folder_name)


    def __get_dataset_for_identicitaion(self):
        
        if stt.sel_method == stt.Method.TRAIN:
            self.__block_num = stt.BLOCK_NUM - const.TRAIN_TEST_SPLIT_VALUE
            self.__dataset, _ = self.__dataset.create_train_dataset_for_identification()
        
        if stt.sel_method == stt.Method.EVALUATE:
            self.__block_num = const.TRAIN_TEST_SPLIT_VALUE
            self.__dataset, _ = self.__dataset.create_test_dataset_for_identification()


    def __create_users_list(self):

        if stt.sel_dataset == stt.Dataset.BALABIT:
            self.__users = ['user12', 'user15', 'user16', 'user20', 'user21', 'user23', 'user29', 'user35', 'user7', 'user9']

        if stt.sel_dataset == stt.Dataset.DFL:
            self.__users = ['User1', 'User10', 'User11', 'User12', 'User13', 'User14', 'User15', 'User16', 'User17', 'User18', 'User19', 'User2',
                'User20', 'User21', 'User3', 'User4', 'User5', 'User6', 'User7', 'User8', 'User9']


    def save_preprocessed_dataset(self, arg):
        arg = arg.value
        switcher = { 
            1: self.__save_preprocessed_identification_dataset_vx,
            2: self.__save_preprocessed_identification_dataset_vy,
            3: self.__save_preprocessed_identification_dataset_vx_vy
        } 
        
        func = switcher.get(arg, lambda: "Wrong evaluation metric!")
        return func()   


    def __save_preprocessed_identification_dataset_vx_vy(self):
        file_name = self.__saved_values_folder_name + '/' + str(stt.sel_dataset) + '_' +  str(stt.sel_method) + '_' \
                        + str(stt.sel_scaling_method) + "_vx_vy_" + str(const.STATELESS_TIME) + "s_stateless_time.csv"

        pos = -1
        id = -1

        f = open(file_name, "w")
        for data in self.__dataset:
            pos = pos + 1

            if pos % self.__block_num == 0:
                id = id + 1
            
            for i in range(0, const.BLOCK_SIZE):
                f.write(str(data[i][0]) + ',')

            for i in range(0, const.BLOCK_SIZE):
                f.write(str(data[i][1]) + ',')

            f.write(self.__users[id] + '\n')

        f.close()


    def __save_preprocessed_identification_dataset_vx(self):
        file_name = self.__saved_values_folder_name + '/' + str(stt.sel_dataset) + '_' +  str(stt.sel_method) + '_' \
                        + str(stt.sel_scaling_method) + "_vx_" + str(const.STATELESS_TIME) + "s_stateless_time.csv"

        pos = -1
        id = -1

        f = open(file_name, "w")
        for data in self.__dataset:
            pos = pos + 1

            if pos % self.__block_num == 0:
                id = id + 1
            
            for i in range(0, const.BLOCK_SIZE):
                f.write(str(data[i][0]) + ',')

            f.write(self.__users[id] + '\n')

        f.close()


    def __save_preprocessed_identification_dataset_vy(self):
        file_name = self.__saved_values_folder_name + '/' + str(stt.sel_dataset) + '_' +  str(stt.sel_method) + '_' \
                        + str(stt.sel_scaling_method) + "_vy_" + str(const.STATELESS_TIME) + "s_stateless_time.csv"

        pos = -1
        id = -1

        f = open(file_name, "w")
        for data in self.__dataset:
            pos = pos + 1

            if pos % self.__block_num == 0:
                id = id + 1
            
            for i in range(0, const.BLOCK_SIZE):
                f.write(str(data[i][1]) + ',')

            f.write(self.__users[id] + '\n')

        f.close()



if __name__ == "__main__":
    gen_csv = GenerateCSV()
    gen_csv.save_preprocessed_dataset(SaveValues.VX)
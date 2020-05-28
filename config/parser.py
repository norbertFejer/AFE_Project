import xml.etree.ElementTree as ET
from enum import Enum

import config.settings as stt
import config.constants as const


class Parser:
    __instance = None


    @staticmethod 
    def getInstance():
        """ Static access method. 
        """
        if Parser.__instance == None:
            Parser()
        return Parser.__instance


    def __init__(self):

        if Parser.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Parser.__instance = self

        self.__tree = ET.parse(const.CONFIG_XML_FILE_LOCATION)
        self.__root = self.__tree.getroot()
        self.__action_num = 0


    def has_next_action(self):
        return self.__action_num < len(self.__root)


    def execute_next_action(self):
        root_num = self.__action_num

        settings = self.__root[root_num].find('settings')
        
        if settings != None:
            for value in settings:
                stt.setter(value.get('name').strip(), value.text.strip().split('.'))

        constants = self.__root[root_num].find('constants')

        if constants != None:
            for value in constants:
                const.setter(value.get('name').strip(), value.text.strip().split('.'))

        self.__action_num = self.__action_num + 1
        


if __name__ == "__main__":
    parser = Parser()
    print('init:')
    print(stt.sel_method)
    print(stt.sel_raw_feature_type)
    print(const.USER_NAME)
    print(const.BLOCK_SIZE)

    print('elso')
    parser.execute_next_action()
    print(stt.sel_method)
    print(stt.sel_raw_feature_type)
    print(const.USER_NAME)
    print(const.BLOCK_SIZE)

    print('masodik')
    parser.execute_next_action()
    print(stt.sel_method)
    print(stt.sel_raw_feature_type)
    print(const.USER_NAME)
    print(const.BLOCK_SIZE)
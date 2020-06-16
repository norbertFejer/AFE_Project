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

        # The entire config.xml content
        self.__tree = ET.parse(const.CONFIG_XML_FILE_LOCATION)
        # XML file root
        self.__root = self.__tree.getroot()
        # Defines how many actions were set 
        self.__action_num = 0


    def has_next_action(self):
        """ Returns True if there is more <action> tag, False otherwise

            Parameters:
                None

            Returns:
                Boolean
        """
        return self.__action_num < len(self.__root)


    def execute_next_action(self):
        """ Executes the next action defined in the <action> tag

            Parameters:
                None

            Returns:
                None
        """ 
        root_num = self.__action_num

        settings = self.__root[root_num].find('settings')
        
        # Sets all values from the <settings> tag
        if settings != None:
            for value in settings:
                stt.setter(value.get('name').strip(), value.text.strip().split('.'))

        constants = self.__root[root_num].find('constants')

        # Sets all values from the <constants> tag
        if constants != None:
            for value in constants:
                const.setter(value.get('name').strip(), value.text.strip().split('.'))

        self.__action_num = self.__action_num + 1
        


if __name__ == "__main__":
    # Testing module functionalities
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
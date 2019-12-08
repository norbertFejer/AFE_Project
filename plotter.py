import dataset as dset
import constants as const

import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter:

    
    def __init__(self):
        self.dataset = dset.Dataset()


    def plot_user_raw_blocks(self):

        df = self.dataset.get_user_raw_data(const.USER_NAME)
        x_standard, y_standard = self.parse_standard_resolution(df['x'].max(), y_max = df['y'].max())
        
        # Calculates the difference between two consecutive row
        tmp_df = df.diff()
        tmp_df = tmp_df.rename(columns={'x': 'dx', 'y': 'dy', 'client timestamp': 'dt'})

        # Gets the index values, where dt is greater than STATELESS_TIME
        outlays_index_list = np.where(tmp_df['dt'] > const.STATELESS_TIME)[0]
        # -1 value is a pivot for using later in the code
        outlays_index_list = np.append([outlays_index_list], [-1])

        # Vectorization for better performance
        tmp_df = tmp_df.values
        ind = 0

        if not os.path.exists(const.SAVED_IMAGES_PATH):
            os.makedirs(const.SAVED_IMAGES_PATH)

        loop_end = int(df.shape[0] / const.BLOCK_SIZE)
        for i in range(loop_end):
            row_start = i * const.BLOCK_SIZE
            row_end = row_start + const.BLOCK_SIZE - 1

            plt.figure()
            while row_start < row_end:
                tmp_row_end = row_end

                if tmp_row_end > outlays_index_list[ind]:
                    tmp_row_end = outlays_index_list[ind]
                    ind = ind + 1
                    
                plt.plot(df['x'][row_start:tmp_row_end], df['y'][row_start:tmp_row_end], marker='o', linestyle='-', markersize=2)
                row_start = tmp_row_end
                    
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.axis([0, x_standard, 0, y_standard])

            name = const.SAVED_IMAGES_PATH + '/' + str(i) + '_block.png'
            plt.savefig(name)


    def parse_standard_resolution(self, x_max, y_max):
        x_standard = [320, 360, 480, 720, 768, 1024, 1280, 1360, 1366, 1440, 1600, 1680, 1920]
        y_standard = [480, 568, 640, 720, 768, 800, 900, 1024, 1050, 1080, 1200, 1280]

        x_pos = 0
        while x_standard[x_pos] < x_max and x_pos < len(x_standard) - 1:
            x_pos += 1

        y_pos = 0
        while y_standard[y_pos] < y_max and y_pos < len(y_standard) - 1:
            y_pos += 1

        if x_max > x_standard[x_pos]:
            x_standard[x_pos] = x_max

        if y_max > y_standard[y_pos]:
            y_standard[y_pos] = y_max

        return x_standard[x_pos], y_standard[y_pos]


    def plot_coordinates(self, x_data, y_data, x_max, y_max):
        plt.plot(x_data, y_data)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.axis([0, x_max, 0, y_max])
        plt.show()

   
if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_user_raw_blocks()
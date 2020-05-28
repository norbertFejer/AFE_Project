import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

import src.dataset as dset
import config.constants as const


class Plotter:

    
    def __init__(self):
        self.dataset = dset.Dataset.getInstance()


    def plot_user_raw_blocks(self):

        df = self.dataset.get_user_raw_data(const.USER_NAME)[['x', 'y', 'client timestamp']]
        x_standard, y_standard = self.parse_standard_resolution(df['x'].max(), y_max = df['y'].max())
        
        # Calculates the difference between two consecutive row
        tmp_df = df.diff()

        # The first row values are NaN, because of using diff() 
        tmp_df = tmp_df[1:].rename(columns={'x': 'dx', 'y': 'dy', 'client timestamp': 'dt'})

        default_time = 0.016
        # Setting default value if dt == 0
        tmp_df.loc[ tmp_df['dt'] <= 0.01, 'dt' ] = default_time

        tmp_df = tmp_df.reset_index(drop=True)
        outlays_index_list = np.concatenate(([0], tmp_df.loc[ tmp_df['dt'] > const.STATELESS_TIME ].index), axis=0)

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
                    
            plt.xlabel('X koordináta')
            plt.ylabel('Y koordináta')
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

    
    def plot_raw_session(self, session_name):
        df = self.dataset.get_session_from_user(session_name)
        print(df.shape)
        df = df[15000:25000]
        df['client timestamp'] -= df['client timestamp'].iloc[0]
        print(df.shape)
        
        plt.figure(figsize=(14,6))
        plt.title("X és Y irányú nyers adatpontok időbeli ábrázolása user7 esetén", fontsize=30)

        sns.lineplot(x="client timestamp", y="x", data=df, label='X koordináta')
        sns.lineplot(x="client timestamp", y="y", data=df, label='Y koordináta')

        plt.xlabel("Adatpontok (idő)", fontsize=30)
        plt.ylabel("Koordináta", fontsize=30)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.legend(fontsize=28)
        plt.show()


    def plot_state_wise_distribution(self):
        df = self.dataset.get_user_raw_data('user7')
        print(df.head(10))

        plt.title("Állapot típus szerinti X koordináta értéke user7 esetén", fontsize=40)

        sns.boxplot(x="x", y="state", data=df)
        plt.xlabel("X koordináta", fontsize=38)
        plt.ylabel("Állapot típus", fontsize=38)
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
        plt.show()
   

if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_user_raw_blocks()
    #plotter.plot_raw_session('C:/Anaconda projects/Software_mod/MouseDynamics/training_files/user7/session_9017095287.csv')
    #plotter.plot_state_wise_distribution()
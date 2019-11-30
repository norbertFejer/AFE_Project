import dataset as dset
import constants as const

import matplotlib.pyplot as plt

class Plotter:

    
    def __init__(self):
        self.dataset = dset.Dataset()


    def plot_user_blocks(self):

        dataset = self.dataset.get_user_preprocessed_dataset(const.USER_NAME)
        print(dataset.head())
        #x_standard, y_standard = self.parse_standard_resolution(df['x'].max(), y_max = df['y'].max())

        # self.plot_coordinates(dataset[0], x_standard, y_standard)


    def parse_standard_resolution(self, x_max, y_max):
        x_standard = [320, 360, 480, 720, 768, 1024, 1280, 1360, 1366, 1440, 1600, 1680, 1920]
        y_standard = [480, 568, 640, 720, 768, 800, 900, 1024, 1050, 1080, 1200, 1280]

        x_pos = 0
        while x_standard[x_pos] < x_max and x_pos < len(x_standard):
            x_pos += 1

        y_pos = 0
        while y_standard[y_pos] < y_max and y_pos < len(y_standard):
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
    plotter.plot_user_blocks()
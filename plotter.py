import dataset as dset
import constants as const

import matplotlib.pyplot as plt

class Plotter:

    
    def __init__(self):
        self.dataset = dset.Dataset()

    def plot_user_blocks(self):
        df = self.dataset.get_dataset_for_user(const.USER_NAME)

        x_max = df['x'].max()
        y_max = df['y'].max()

        x_standard, y_standard = self.parse_standard_resolution(x_max, y_max)
        print(x_standard)
        print(y_standard)


    def parse_standard_resolution(self, x_max, y_max):
        x_standard = [320, 360, 480, 720, 768, 1024, 1280, 1360, 1366, 1440, 1600, 1680, 1920]
        y_standard = [480, 568, 640, 720, 768, 800, 900, 1024, 1050, 1080, 1200, 1280]

        x_pos = 0
        while x_standard[x_pos] < x_max and x_pos < len(x_standard):
            x_pos += 1

        y_pos = 0
        while y_standard[y_pos] < y_max and y_pos < len(y_standard):
            y_pos += 1

        return x_standard[x_pos], y_standard[y_pos]

   
if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_user_blocks()
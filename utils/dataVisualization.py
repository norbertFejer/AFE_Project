import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import src.dataset as dset
import config.constants as const
import config.settings as stt


dataset = dset.Dataset.getInstance()

def plot_user_normalized_data():
    stt.sel_authentication_type = stt.AuthenticationType.ONE_CLASS_CLASSIFICATION
    stt.sel_occ_features = stt.OCCFeatures.RAW_X_Y_DIR
    stt.sel_user_recognition_type = stt.UserRecognitionType.AUTHENTICATION

    X_train, _ = dataset.create_train_dataset_for_authentication(const.USER_NAME)
    df = pd.DataFrame({'x': X_train[:, 0], 'y': X_train[:, 1]})

    print('max x ', df['x'].max())
    print('max y ', df['y'].max())
    print('min x ', df['x'].min())
    print('min y ', df['y'].min())
    
    # for i in range(int(df.shape[0] / const.BATCH_SIZE)):
    #     start_pos = i * const.BLOCK_SIZE
    #     stop_pos = start_pos + const.BLOCK_SIZE
    #     sns.lineplot(data=df['x'][start_pos : stop_pos])

    # plt.show()


if __name__ == "__main__":
    plot_user_normalized_data()
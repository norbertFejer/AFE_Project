import pandas as pd
import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd 

import src.dataset as dset
import config.constants as const
import config.settings as stt


dataset = dset.Dataset.getInstance()

def get_user_raw_dataset(username, concat_chuncks_value):
    stt.sel_chunck_samples_handler = concat_chuncks_value
    return dataset.get_user_all_preprocessed_data(username)


def plot_data_distribution():
    users = stt.get_balabit_users()

    drop_chunck_samples = np.array([]).astype(int)
    concatenate_chunck_samples = np.array([]).astype(int)

    for user in users:
        print(user)
        dataset = get_user_raw_dataset(user, stt.ChunkSamplesHandler.DROP_CHUNKS)
        drop_chunck_samples = np.append(drop_chunck_samples, dataset.shape[0])

        dataset = get_user_raw_dataset(user, stt.ChunkSamplesHandler.CONCATENATE_CHUNKS)
        concatenate_chunck_samples = np.append(concatenate_chunck_samples, dataset.shape[0])

    df = pd.DataFrame({'username':users, 'drop_chunck_samples':drop_chunck_samples , 'concatenate_chunck_samples':concatenate_chunck_samples })
    my_range=range(1, len(df.index) + 1)

    plt.hlines(y=my_range, xmin=df['drop_chunck_samples'], xmax=df['concatenate_chunck_samples'], color='grey', alpha=0.6)
    plt.scatter(df['drop_chunck_samples'], my_range, color='skyblue', alpha=1, label='Csak kompakt blokkokat tartalmaz', s=100)

    for x, y in zip(drop_chunck_samples, my_range):

        label = x
        plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(-44, -8), # distance from text to points (x,y)
                        ha='center',
                        fontsize=20,
                        fontweight='bold') # horizontal alignment can be left, right or center

    for x, y in zip(concatenate_chunck_samples, my_range):

        label = x
        plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(44, -8), # distance from text to points (x,y)
                        ha='center',
                        fontsize=20,
                        fontweight='bold') # horizontal alignment can be left, right or center


    plt.scatter(df['concatenate_chunck_samples'], my_range, color='green', alpha=0.6 , label='Kompakt és töredezett blokkokat is tartalmaz', s=100)

    plt.legend()
    
    # Add title and axis names
    plt.yticks(my_range, df['username'])
    plt.title("Blokk darabszám BALABIT adathalmaz esetén", loc='center', fontsize=40)
    plt.xlabel('Blokk darabszám', fontsize=30)
    plt.ylabel('Felhasználó', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=24)
    plt.xscale('log')
    plt.show()


def plot_dataset_comparision():
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    balabit_samples = [426060, 415668, 246824, 149921, 240046, 294197, 125931, 124548, 131751, 98870]
    dfl_samples = [2682238, 1482569, 979159, 41355248, 14582010, 3894676, 3960115, 687581, 464349, 474376, 4676577, 5906237, 2850973, 2580515, 12578677, 2007739, 2489803, 25213286, 16107993, 2788726, 3806959]

    num_bars = len(balabit_samples)
    x_pos = [25] * num_bars
    y_pos = np.arange(0, 2 * num_bars, 2)
    z_pos = [0] * num_bars
    x_size = [1] * num_bars
    y_size = [1] * num_bars
    z_size = balabit_samples

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color='aqua')

    num_bars2 = len(dfl_samples)
    x_pos2 = [15] * num_bars2
    y_pos2 = np.arange(0, 2 * num_bars2, 2)
    z_pos2 = [0] * num_bars2
    x_size2 = [1] * num_bars2
    y_size2 = [1] * num_bars2
    z_size2 = dfl_samples

    ax.bar3d(x_pos2, y_pos2, z_pos2, x_size2, y_size2, z_size2, color='blue')

    ax.set_xlim([0,41])

    for x, y, z in zip(x_pos, y_pos, z_size):
        label = 'user15'
        ax.text(x, y, z, label, (0,0,0))

    for x, y, z in zip(x_pos2, y_pos2, z_size2):
        label = 'user15'
        ax.text(x, y, z, label, (0,0,0))

    plt.show()


def print_velocity_distribution():
    df = dataset.get_user_preprocessed_dataset('user7')
    df = df.reshape((38400, 2))

    plt.figure(figsize=(20,10))
    plt.title('X irányú sebesség eloszlása', fontsize=30)
    sns.distplot(df[:, 0], hist=True)

    plt.xlabel('Sebesség (px/s)', fontsize=30)
    plt.ylabel('Normalizált valószínűségi sűrűség', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.show()


def print_velocity_distribution_both_axis():
    df = dataset.get_user_preprocessed_dataset('user9')
    df = df.reshape((38400, 2))

    #x = np.random.randint(1,100,100)
    #y = np.random.randint(1,100,100)
    #sns.kdeplot(df[:, 0], df[:, 1], shade=True, cmap="Reds")

    rc={'axes.labelsize': 32, 'xtick.labelsize': 32, 'ytick.labelsize': 32}
    sns.set(rc=rc)

    h = sns.jointplot(x=df[:, 0], y=df[:, 1], kind="kde")
    h.set_axis_labels('x irányú sebesség (px/s)', 'y irányú sebesség (px/s)')
    plt.show()


def print_normalized_dataset():
    trainX, testX = dataset.create_train_dataset_for_authentication('user7')
    trainX = trainX[10:15, :, 0]
    
    for i in range(trainX.shape[0]):
        plt.plot(trainX[i])

    plt.title('X irányú normalizált sebesség 128-as ablakmérettel', fontsize=30)
    plt.ylabel('X irányú sebesség (px/s)', fontsize=30)
    plt.xlabel('Idő (s)', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.show()


def plot_block_size_evaluation_results():
    acc_value = [0.647887324, 0.687323944, 0.756338028, 0.815915493, 0.76028169, 0.7, 0.649295775]
    block_size = [16, 32, 64, 128, 256, 512, 1024]

    df = pd.DataFrame(list(zip(acc_value, block_size)), columns =['acc_value', 'block_size']) 

    g = sns.barplot(x=block_size, y=acc_value, palette=("Blues_d"), data=df)

    for index, row in df.iterrows():
        g.text(row.name, row.acc_value + 0.02, round(row.acc_value, 2), color='black', ha="center", fontsize=20, fontweight='bold')

    plt.title('Balabit teszt adathalmazra mért MCDCNN pontossága', fontsize=30)
    plt.ylabel('ACC érték', fontsize=30)
    plt.xlabel('Blokkméret (adatpont)', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.show()


def plot_occ_balabit_results_boxplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/occ_1000_balabit_p.csv")

    sns.boxplot(data=df, linewidth=2)

    plt.title('OCSVM eredménye Balabit adathalmaz esetén', fontsize=30)
    plt.ylabel('AUC érték', fontsize=30)
    plt.xlabel('Tanító adathalmaz típusa', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=28)
    plt.ylim(0.3, 1)    
    plt.show()


def plot_occ_dfl_results_boxplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/occ_300_dfl_p.csv")

    sns.boxplot(data=df, linewidth=2)

    plt.title('OCSVM eredménye DFL adathalmaz esetén', fontsize=30)
    plt.ylabel('AUC érték', fontsize=30)
    plt.xlabel('Tanító adathalmaz típusa', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=28)
    plt.ylim(0.3, 1) 
    plt.show()


def plot_binary_classification_balabit_results_boxplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/binary_300_balabit_p.csv")

    sns.catplot(x="model", y="value", hue="metric", kind="box", data=df, legend=False, linewidth=2)
    sns.despine(top=False, right=False)

    plt.title('Bináris osztályozás eredménye Balabit adathalmaz esetén', fontsize=30)
    plt.ylabel('Érték', fontsize=30)
    plt.xlabel('CNN modell típusa', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=28)
    plt.ylim(0.35, 1) 
    plt.legend(fontsize=28, loc='lower right')
    plt.show()


def plot_binary_classification_dfl_results_boxplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/binary_440_dfl_p.csv")

    sns.catplot(x="model", y="value", hue="metric", kind="box", data=df, legend=False, linewidth=2)
    sns.despine(top=False, right=False)

    plt.title('Bináris osztályozás eredménye DFL adathalmaz esetén', fontsize=30)
    plt.ylabel('Érték', fontsize=30)
    plt.xlabel('CNN modell típusa', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=28)
    plt.ylim(0.35, 1) 
    plt.legend(fontsize=28, loc='lower right')
    plt.show()


def plot_occ_balabit_results_barplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/occ_balabit_barplot.csv")

    sns.catplot(x="model", y="value", hue='metric', data=df, kind="bar", legend=False, linewidth=2.5,
                 errcolor=".2", edgecolor=".2") 
    sns.despine(top=False, right=False)

    plt.title('OCSVM átlag AUC értéke Balabit adathalmaz esetén', fontsize=30)
    plt.ylabel('AUC érték', fontsize=30)
    plt.xlabel('Tanító adathalmaz típusa', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.55, 0.60, 0.65, 0.70, 0.75], fontsize=28)
    plt.ylim(0.55, 0.75)
    plt.legend(fontsize=28, loc='upper right')
    plt.show()


def plot_occ_dfl_results_barplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/occ_dfl_barplot.csv")

    sns.catplot(x="model", y="value", hue='blokkszam', data=df, kind="bar", legend=False, linewidth=2.5,
                 errcolor=".2", edgecolor=".2") 
    sns.despine(top=False, right=False)

    plt.title('OCSVM átlag AUC értéke DFL adathalmaz esetén', fontsize=30)
    plt.ylabel('AUC érték', fontsize=30)
    plt.xlabel('Tanító adathalmaz', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.55, 0.60, 0.65, 0.70, 0.75], fontsize=28)
    plt.ylim(0.55, 0.75)
    plt.legend(fontsize=28, loc='upper right')
    plt.show()


def plot_binary_classification_balabit_results_barplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/binary_classification_balabit_barplot.csv")

    sns.catplot(x="model", y="value", hue='metric', data=df, kind="bar", legend=False, linewidth=2.5,
                 errcolor=".2", edgecolor=".2") 
    sns.despine(top=False, right=False)

    plt.title('Bináris osztályozás átlag eredménye Balabit adathalmaz esetén', fontsize=30)
    plt.ylabel('Érték', fontsize=30)
    plt.xlabel('CNN modell típusa', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.45, 0.55, 0.65, 0.75, 0.85, 0.95], fontsize=28)
    plt.ylim(0.45, 1)
    plt.legend(fontsize=28, loc='upper center')
    plt.show()


def plot_binary_classification_dfl_results_barplot():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/binary_classification_dfl_barplot.csv")

    sns.catplot(x="model", y="value", hue='metric', data=df, kind="bar", legend=False, linewidth=2.5,
                 errcolor=".2", edgecolor=".2") 
    sns.despine(top=False, right=False)

    plt.title('Bináris osztályozás átlag eredménye DFL adathalmaz esetén', fontsize=30)
    plt.ylabel('Érték', fontsize=30)
    plt.xlabel('CNN modell típusa', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.45, 0.55, 0.65, 0.75, 0.85, 0.95], fontsize=28)
    plt.ylim(0.45, 1)
    plt.legend(fontsize=28, loc='upper center')
    plt.show()


def plot_occ_aggregated_blocks_result():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/occ_agg_blocks_res.csv")

    sns.lineplot(y="value", x="block_num", hue="modell", data=df)

    plt.title('Aggregált blokkokal mért eredmény Balabit adathalmazon', fontsize=30)
    plt.ylabel('Átlag AUC', fontsize=30)
    plt.xlabel('Aggregált blokkok száma', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28, loc='lower right')
    plt.show()


def plot_binary_classification_aggregated_blocks_result():
    df = pd.read_csv("C:/Anaconda projects/Software_mod/evaluationResults/binary_classification_agg_blocks_res.csv")

    sns.lineplot(y="value", x="block_num", hue="modell", data=df)

    plt.title('Aggregált blokkokal mért eredmény Balabit adathalmazon', fontsize=30)
    plt.ylabel('Átlag AUC', fontsize=30)
    plt.xlabel('Aggregált blokkok száma', fontsize=30)
    plt.xticks([1, 2, 3, 4, 5], fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28, loc='center right')
    plt.show()


if __name__ == "__main__":
    #plot_data_distribution()
    #plot_dataset_comparision()
    #print_velocity_distribution()
    #print_velocity_distribution_both_axis()
    #print_normalized_dataset()
    #plot_block_size_evaluation_results()

    #plot_occ_balabit_results_boxplot()
    plot_occ_dfl_results_boxplot()
    #plot_binary_classification_balabit_results_boxplot()
    #plot_binary_classification_dfl_results_boxplot()
    #plot_occ_balabit_results_barplot()
    #plot_occ_dfl_results_barplot()
    #plot_binary_classification_balabit_results_barplot()
    #plot_binary_classification_dfl_results_barplot()
    #plot_occ_aggregated_blocks_result()
    #plot_binary_classification_aggregated_blocks_result()
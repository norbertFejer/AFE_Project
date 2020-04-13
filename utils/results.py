import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import PathPatch


def plt_1():

    data1 = [64, 51.1428571, 67]
    data2 = [58.7142857, 37.5714286, 59.8571429]
    data = [64, 51.14, 67, 58.71, 37.57, 59.85]
    width = 0.6
    x1 = [0, 1, 2]
    x2 = [4, 5, 6]

    p1 = plt.bar(x1, data1, width=width)
    p2 = plt.bar(x2, data2, width=width)
    plt.ylim(0, 80)
    labels = ['plain', 'transfer', 'update', 'plain', 'transfer', 'update']
    x = [0, 1, 2, 4, 5, 6]
    plt.xticks(x, labels, rotation=15)
    plt.ylabel('Accuracy (%)', size=14)
    plt.legend((p1[0], p2[0]), ('300 training samples', 'All training samples'))
    plt.title('Identification results on Balabit Dataset', size=16)

    for i in range(len(x)):
        plt.annotate(str(data[i]), xy=(x[i], data[i] + 1), ha='center', size=12)

    plt.show()


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def plt_2():

    fig = plt.figure()
    df = pd.read_csv('C:/Anaconda projects/Software_mod/evaluationResults/res_aut_300.csv')
    bp = sns.boxplot(x="type", y="value", hue="metric", data=df)
    bp.set_xlabel('Measurement type', size=14)
    bp.set_ylabel('Value', size=14)
    bp.tick_params(labelsize=12)
    adjust_box_widths(fig, 0.9)
    plt.title('Authentication result using 300 training samples', size=14)
    plt.show()


def plt_3():

    fig = plt.figure()
    df = pd.read_csv('C:/Anaconda projects/Software_mod/evaluationResults/res_aut_test.csv')
    bp = sns.boxplot(x="type", y="value", hue="metric", data=df)
    bp.set_xlabel('Measurement type', size=14)
    bp.set_ylabel('Value', size=14)
    bp.tick_params(labelsize=12)
    adjust_box_widths(fig, 0.9)
    plt.title('Authentication result using all training samples', size=14)
    plt.show()


if __name__ == "__main__":
    plt_3()


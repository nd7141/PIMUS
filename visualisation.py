'''

'''
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

__author__ = 'sivanov'

def get_coords(*filenames):
    x_lst = []
    y_lst = []
    for filen in filenames:
        with open(filen) as f:
            d = zip(*[line.split() for line in f])
            x_lst.append(map(int, d[0]))
            y_lst.append(map(float, d[1]))
    return x_lst, y_lst

def plot_double(x_lst, y_lst, legends, xlabels, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    axi = [ax1, ax2]
    plots = []
    colors = ['r', 'b']

    for i in range(len(x_lst)):
        p, = axi[i].plot(x_lst[i], y_lst[i], linewidth=3, color=colors[i])
        axi[i].set_xlabel(xlabels[i])
        plots.append(p)

    plt.legend(plots, legends, loc=4)

    plt.show()
    fig.savefig(filename, dpi=fig.dpi)

if __name__ == "__main__":
    console = []
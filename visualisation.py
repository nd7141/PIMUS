'''

'''
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

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

def plot_double(x, y_lst, legends, xlabels, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    axi = [ax1, ax2]
    plots = []
    colors = ['r', 'b']

    for i in range(len(y_lst)):
        p, = axi[i].plot(x, y_lst[i], linewidth=3, color=colors[i])
        axi[i].set_xlabel(xlabels[i])
        plots.append(p)

    plt.legend(plots, legends, loc=4)

    plt.show()
    fig.savefig(filename, dpi=fig.dpi)

def visualiseResults(x, y_lst, legends, xlabel="", ylabel="", title="", filename="",):
    matplotlib.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(18, 10))

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # plt.yscale('log')

    colors = ['b', 'r', 'g', 'm', 'k', u'#abfeaa', u'#cccabc', u'#1111ee', 'y', 'c', u'#fe2fb3']
    marks = ["o", "s", "^", "v", 'x', "<", ">", '8', "<", ">", '8']
    colors = colors[::1]
    marks = marks[::1]
    y_lst.reverse()
    legends.reverse()
    colors.reverse()
    marks.reverse()

    plots = []
    # print colors
    for i in range(len(y_lst)):
        plt.plot(x, y_lst[i], color=colors[i], linewidth=3)
        p, = plt.plot(x, y_lst[i], color = colors[i], marker = marks[i], markersize=10)
        plots.append(p)

    # plt.xlim([9, 200])
    # plt.ylim([770, 820])

    plt.legend(plots, legends, loc=2, prop={'size': 24})
    plt.grid()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if title:
        plt.title('%s' %(title), fontsize = 24)
    plt.show()
    # if os.path.exists(filename):
    #     os.remove(filename)
    if filename:
        fig.savefig(filename, dpi=fig.dpi)

if __name__ == "__main__":
    x = range(5, 55, 5)
    with open("datasets/gnutella_results.txt") as f:
        eu = []
        tope = []
        topn = []
        for line in f:
            d = map(float, line.split())
            eu.append(d[0])
            tope.append(d[1])
            topn.append(d[2])

    visualiseResults(x, [eu, tope, topn], ['EU', 'Top-Edges', 'Top-Nodes'], filename="datasets/gnutella_results.png")

    console = []
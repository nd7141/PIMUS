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
    # plt.ylim([88, 152])

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

def bar_plot(y, xticks, xlabel="", ylabel="", filename="", title=""):
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 2

    x = np.arange(0, 4*len(y), 4)
    colors = ['r', 'g', 'm', 'k', 'y', 'c', u'#fe2fb3', u'#abfeaa', u'#cccabc', u'#1111ee', 'b']
    colors = ['k', 'k']
    colors = colors[::1]

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel, fontsize = 24)

    if title:
        plt.title('%s' %(title), fontsize = 18)

    plt.grid(axis="y", linestyle='-', linewidth=2)



    rects1 = ax.bar(x, y, width = 3, bottom=0, color = "k", log=True)
    plt.xticks(x + 1.5, xticks, fontsize=17)

    # add text label at the top of each bar
    # solution found at http://matplotlib.org/examples/api/barchart_demo.html
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.1f'%height,
                    ha='center', va='bottom')

    autolabel(rects1)
    for i, rect in enumerate(rects1):
        rect.set_color(colors[i])
    if filename:
        fig.set_size_inches(15.5,11.5)
        fig.savefig(filename, dpi=fig.dpi)
    plt.show()

if __name__ == "__main__":
    x = range(5, 55, 5)
    with open("datasets/gnutella_results2_mv_greedy.txt") as f:
        d = map(float, f.readlines()[0].split())
        gr = d[1::2]
    # print gr
    with open("datasets/gnutella_results2_mv.txt") as f:
        eu = []
        tope = []
        topn = []
        for line in f:
            d = map(float, line.split())
            eu.append(d[1])
            tope.append(d[2])
            topn.append(d[3])

    visualiseResults(x, [eu, tope, topn, gr], ['Explore-Update', 'Top-Edges', 'Top-Nodes', 'Greedy'], xlabel='Number of features in F, K',
                     ylabel='Influence Spread', filename="datasets/gnutella_results2_mv2.png")

    # with open("datasets/gnutella_time2_mv.txt") as f:
    #     y = map(float, f.readlines()[0].split())
    #
    # with open("datasets/gnutella_time2_mv_greedy.txt") as f:
    #     y.append(float(f.readlines()[0]))
    #
    # bar_plot(y, ['Explore-Update', 'Greedy'], xlabel='Algorithms', ylabel='Execution time (sec)', filename="datasets/gnutella_time2_mv.png")

    console = []
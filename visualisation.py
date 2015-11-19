'''

'''
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import spline

__author__ = 'sivanov'

# These are the "Tableau 20" colors as RGB.
tableau20 = [u'#1f77b4', u'#aec7e8', u'#ff7f0e', u'#ffbb78', u'#2ca02c', u'#98df8a', u'#d62728', u'#ff9896', u'#9467bd',
              u'#c5b0d5', u'#8c564b', u'#c49c94', u'#e377c2', u'#f7b6d2', u'#7f7f7f', u'#c7c7c7', u'#bcbd22', u'#dbdb8d',
              u'#17becf', u'#9edae5']

def rgb_to_hex(rgb):
    return u'#%02x%02x%02x' % rgb

tableau_colors = ['#deb0b0', '#b0c4de', u'#98df8a', u'#ff9896', u'#c5b0d5', u'#c49c94', u'#f7b6d2']


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
        plt.plot(x, y_lst[i], color=tableau_colors[i], linewidth=5)
        p, = plt.plot(x, y_lst[i], color = tableau_colors[i], marker = marks[i], markersize=20)
        plots.append(p)

    # plt.xlim([9, 200])
    # plt.ylim([88, 152])

    plt.xticks(range(5, 55, 10))

    plt.legend(plots, legends, loc=2, prop={'size': 35})
    plt.grid()
    plt.ylabel(ylabel, fontsize=40)
    plt.xlabel(xlabel, fontsize=40)
    if title:
        plt.title('%s' %(title), fontsize = 48)
    plt.tight_layout()
    plt.show()
    # if os.path.exists(filename):
    #     os.remove(filename)
    if filename:
        fig.savefig(filename, dpi=1200)

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

    plt.tight_layout()
    plt.show()

def double_axis_plot(x, y_lst1, y_lst2, title="", fontsize=20, filename="", legends=[], linewidth=5, xlabel="", smoothed=False):
    matplotlib.rcParams.update({'font.size': fontsize})
    fig, ax1 = plt.subplots(figsize=(18, 10))
    colors = ['b', 'r', 'g']
    marks = ["o", "s", "^"]

    plots = []
    # print colors
    for i in range(len(y_lst1)):
        ax1.plot(x, y_lst1[i], color=tableau_colors[i], linewidth=linewidth)
        p, = ax1.plot(x, y_lst1[i], color = tableau_colors[i], marker = marks[i], markersize=20)
        plots.append(p)

    if xlabel:
        ax1.set_xlabel(xlabel)
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Influence spread', fontsize=fontsize)
    # for tl in ax1.get_yticklabels():
    #     tl.set_color('b')

    ax1.grid()
    # plt.xticks([21, 31, 41, 81, 91, 101])
    if title:
        plt.title(title, fontsize=fontsize)


    ax2 = ax1.twinx()
    for i in range(len(y_lst2)):
        ax2.plot(x, y_lst2[i], color=tableau_colors[2], linewidth=linewidth)
        p, = ax2.plot(x, y_lst2[i], color = tableau_colors[2], marker = marks[2], markersize=20)
        plots.append(p)
    ax2.set_ylabel('Running time (sec)', fontsize=fontsize)
    # for tl in ax2.get_yticklabels():
    #     tl.set_color(colors[2])

    # ax2.set_yscale("log")

    if smoothed:
        x_smoothed = np.linspace(min(x), max(x), 300)
        y_smoothed = spline(x, y_lst1[0], x_smoothed)
        ax1.plot(x_smoothed, y_smoothed)



    if legends:
        plt.legend(plots, legends, loc=2, prop={'size': fontsize})

    plt.tight_layout()
    plt.show()

    if filename:
        fig.savefig(filename, dpi=1200)

def two_bar_plots(y1_lst, y2_lst, num=2, xticks=[], legends=[], xlabel="", ylabel="", fontsize=20, filename=""):
    matplotlib.rcParams.update({'font.size': fontsize})

    ind = np.arange(num)
    width = 0.85

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    ax.bar(ind+width+0.45, y2_lst, 0.2, color='#deb0b0', log=True)
    ax.bar(ind+width+0.35, y1_lst, 0.2, color='#b0c4de', log=True)


    ax.set_xticks(ind+width+(width/2))
    if xticks:
        ax.set_xticklabels(xticks)

    if legends:
        ax.legend(legends, loc=2)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)

    plt.grid()

    plt.tight_layout()
    plt.show()
    if filename:
        fig.savefig(filename, dpi=1200)

if __name__ == "__main__":
    model = "wc"
    # x = range(5, 55, 5)
    # with open("datasets/gnutella_results2_mv_greedy.txt") as f:
    #     d = map(float, f.readlines()[0].split())
    #     gr = d[1::2]
    # print gr
    # with open("datasets/experiment1/gnutella_results2_{}.txt".format(model)) as f:
    #     # x = []
    #     eu = []
    #     tope = []
    #     topn = []
    #     greedy = []
    #     for line in f:
    #         d = map(float, line.split())
    #         greedy.append(d[0])
    #         eu.append(d[1])
    #         tope.append(d[2])
    #         topn.append(d[3])

    # with open("datasets/greedy/gnutella_greedy_results_{}.txt".format(model)) as f:
    #     greedy_new = []
    #     for line in f:
    #         greedy_new.append(float(line))

    # with open("datasets/experiment3/gnutella_time_{}.txt".format(model)) as f:
    #     eu_timing = []
    #     for line in f:
    #         eu_timing.append(float(line.split()[1]))

    # visualiseResults(x, [eu, tope, topn, greedy_new], ['Explore-Update', 'Top-Edges', 'Top-Nodes', 'Greedy'], xlabel='Number of selected features, |K|',
    #                  ylabel='Influence Spread', filename="datasets/experiment1/gnutella_results2_{}.pdf".format(model),
    #                  title="")

    # double_axis_plot(x, [eu, tope], [eu_timing], title="Inf. Spread for K = 50.",
    #                  filename="datasets/experiment3/gnutella_results_time_{}.pdf".format(model), legends = ['Explore-Update', 'Top-Edges', 'Runtime of Explore-Update'],
    #                  fontsize=40, linewidth=6)

    # eu_time = []
    # greedy_time = []
    # with open("datasets/experiment1/gnutella_time2_mv.txt") as f:
    #     d = map(float, f.readlines()[0].split())
    #     eu_time.append(d[0])
    #     greedy_time.append(d[1])
    # with open("datasets/experiment1/gnutella_time2_wc.txt") as f:
    #     d = map(float, f.readlines()[0].split())
    #     eu_time.append(d[0])
    #     greedy_time.append(d[1])
    #
    #
    # two_bar_plots(eu_time, greedy_time, num=2, xticks=['Multivalency', 'Weighted Cascade'],
    #               legends=['Greedy', 'Explore-Update'], ylabel='Running time (sec)', fontsize=35,
    #               filename="datasets/experiment1/gnutella_time_both.pdf")



    # experiment 4
    x = [2**i*10 for i in range(6)]
    with open("datasets/experiment4/gnutella_theta_spread.txt") as f1:
        eu_spread = []
        for line in f1:
            eu_spread.append(float(line))
    with open("datasets/experiment4/gnutella_theta_time.txt") as f2:
        eu_time = []
        for line in f2:
            eu_time.append(float(line))

    double_axis_plot(x, [eu_spread], [eu_time], legends=["Spread", "Running time"], title="Influence spread and Runtime \nvs. threshold $\\theta$",
                     smoothed=True)

    # with open("datasets/gnutella_time2_wc.txt") as f:
    #     y = map(float, f.readlines()[0].split())
    #
    # with open("datasets/gnutella_time2_wc_greedy.txt") as f:
    #     y.append(float(f.readlines()[0]))
    #
    # bar_plot(y, ['Explore-Update', 'Greedy'], xlabel='Algorithms', ylabel='Execution time (sec)', filename="datasets/gnutella_time2_wc.png")

    console = []
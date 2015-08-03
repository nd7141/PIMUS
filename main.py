'''

'''
from __future__ import division
import networkx as nx
from greedy import greedy
from heuristics import *
from runMC import run_ic, run_lt
import comparison
__author__ = 'sivanov'


def read_graph(filename, directed=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    old2new = dict()
    count = 0
    with open(filename) as f:
        for line in f:
            d = line.split()
            if int(d[0]) not in old2new:
                old2new[int(d[0])] = count
                count += 1
            if int(d[1]) not in old2new:
                old2new[int(d[1])] = count
                count += 1
            G.add_edge(old2new[int(d[0])], old2new[int(d[1])], weight=float(d[2]))
    return G


def read_likes(filename):
    L = dict()
    with open(filename) as f:
        for line in f:
            d = line.strip().split()
            L[int(d[0])] = d[1:]
    return L

if __name__ == "__main__":

    G = read_graph("datasets/Wiki-Vote_graph_ic.txt", True)
    L = read_likes("datasets/Wiki-Vote_likes.txt")
    K = 30
    R = 100

    #comparison of PIMUS to IM
    # S = random.sample(G, 50)
    # f = greedy(G, L, S, K, R, False)
    # print 'PIMUS spread:', run_lt(G, L, S, f, R)

    # S = comparison.greedy_im(G, 5, R)
    # print 'IM spread:', comparison.run_ic(G, S, R)

    # comparison of different PIMUS algorithms
    l = 20
    count = 0
    spread = 0
    for _ in range(l):
        S = random.sample(G.nodes(), 50)
        f1 = greedy(G, L, S, K, R)
        print sorted(f1)

        spread += run_ic(G, L, S, f1, R)
    print 'Average spread:', spread/l
    with open('data/spread{0}.txt'.format(K), 'w+') as f:
        f.write("{0}".format(spread/l))

    console = []
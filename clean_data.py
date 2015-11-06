# author: sivanov
# date: 06 Nov 2015
from __future__ import division
import networkx as nx
import numpy as np
import math


def convert_idx(filename, output):
    """
    Convert graph file indexing to 0 1 2 ...
    :param filename:
    :param output:
    :return:
    """
    old2new = dict()
    count = 0
    with open(filename) as f:
        with open(output, 'w+') as g:
            for line in f:
                d = line.split()
                u = int(d[0])
                v = int(d[1])
                if u not in old2new:
                    old2new[u] = count
                    count += 1
                if v not in old2new:
                    old2new[v] = count
                    count += 1
                if u != v:
                    g.write('%s %s\n' %(old2new[u], old2new[v]))


def read_graph2(filename, directed=True):
    """
    Create networkx graph reading file.
    :param filename: every line (u, v)
    :param directed: boolean
    :return:
    """
    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            d = line.split()
            G.add_edge(int(d[0]), int(d[1]))
    return G

def wc_model(G):
    P = dict()
    for u in G:
        out_edges = G.out_edges(u)
        d = len(out_edges)
        for e in out_edges:
            P[e] = 1./d
    return P

def mv_model(G, prange):
    P = dict()
    for e in G.edges():
        p = np.random.choice(prange)
        P[e] = p
    return P

def add_weights(G, P, log=True):
    for e in P:
        if log:
            G[e[0]][e[1]]['weight'] = -math.log(P[e])
        else:
            G[e[0]][e[1]]['weight'] = P[e]


if __name__ == "__main__":

    convert_idx('datasets/p2p-Gnutella04.txt', 'datasets/gnutella.txt')
    console = []
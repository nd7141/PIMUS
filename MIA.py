# author: sivanov
# date: 27 Oct 2015
from __future__ import division
import networkx as nx
import pandas as pd
import numpy as np
import time

def read_graph(filename, directed=True, sep=' ', header = None):
    """
    Create networkx graph using pandas.
    :param filename: every line (u, v)
    :param directed: boolean
    :param sep: separator in file
    :return
    """
    df = pd.read_csv(filename, sep=sep, header = header)
    if directed:
        G = nx.from_pandas_dataframe(df, 0, 1, create_using=nx.DiGraph())
    else:
        G = nx.from_pandas_dataframe(df, 0, 1)
    return G

def read_graph2(filename, directed=False):
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

def add_graph_attributes(G, filename):
    """
    Add features as node attributes and construct a Ef
    :param G: networkx graph
    :param filename: (u f1 f2 f3 ...); u is mandatory in graph -- the node
    f1 f2 f3 ... - arbitrary number of features
    :return: Ef: dictionary f -> edges that it affects
    """
    Ef = dict() # feature -> edges
    with open(filename) as f:
        for line in f:
            d = line.split()
            u = int(d[0])
            features = d[1:]
            for f in features:
                Ef.setdefault(f, []).extend(G.in_edges(u)) # add feature-dependent edges
            G.node[u]['Fu'] = features
    return Ef

def read_probabilities(filename, sep=' '):
    """
    Creates a dataframe object indexed by endpoints of edge (u, v)
    :param filename: (u, v, p)
    :param sep: separator in the file
    :return:
    """
    df = pd.read_csv(filename, sep=sep, header = None)
    return df.set_index([0, 1])

def update_probabilities(G, B, Q, F, E, K, P):
    """
    :param G: graph that has nodes attributes as features for nodes
    :param B: dataframe indexed by two endpoints of edge: base probabilities on edges
    :param Q: dataframe indexed by two endpoints of edge: product probabilities on edges
    :param F: selected set of features
    :param E: edges that require update
    :param K: number of required features
    :param P: final probabilities on edges (updated only Ef)
    :return:
    """
    for e in E:
        hF = len(set(F).intersection(G.node[e[1]]['Fu']))/K # function h(F)
        q = float(Q.loc[e])
        b = float(B.loc[e])
        P[e] = hF*q + b # final probabilities p = h(F)*q + b

if __name__ == "__main__":

    G = read_graph('datasets/wv.txt')
    Ef = add_graph_attributes(G, 'datasets/wv_likes.txt')

    B = read_probabilities('datasets/Wiki-Vote_graph_ic.txt')
    Q = read_probabilities('datasets/Wiki-Vote_graph_ic.txt')

    # intialize
    P = dict()
    start = time.time()
    # TODO initialize by reading a file, maybe P making pandas dataframe
    update_probabilities(G, B, Q, [], G.edges(), 1, P)
    print time.time() - start
    print P[(0, 1)]

    # TODO rewrite reading graph using 'datasets/Wiki-Vote_graph_ic.txt' with third parameter. Then add headers. Add to from_pandas third parameter. Second, read filegraph


    # f = np.random.choice(Ef.keys())
    # F = [f]

    console = []
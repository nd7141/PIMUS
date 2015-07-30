'''

'''
from __future__ import division
import networkx as nx
import random
from greedy import greedy
from heuristics import ffc
from runMC import run_mc
__author__ = 'sivanov'


def read_graph(filename, directed=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    with open(filename) as f:
        for line in f:
            d = line.split()
            G.add_edge(int(d[0]), int(d[1]), weight=float(d[2]))
    return G


def read_likes(filename):
    L = dict()
    with open(filename) as f:
        for line in f:
            d = line.strip().split()
            L[int(d[0])] = d[1:]
    return L

if __name__ == "__main__":

    G = read_graph("datasets/Wiki-Vote_graph.txt", True)
    L = read_likes("datasets/Wiki-Vote_likes.txt")
    S = random.sample(G.nodes(), 10)
    K = 5
    R = 100

    f1 = greedy(G, L, S, K, R)
    f2 = ffc(G, L, S, K)
    print sorted(f1)
    print sorted(f2)

    spread1 = run_mc(G, L, S, f1, R)
    spread2 = run_mc(G, L, S, f2, R)

    print spread1, spread2

    console = []
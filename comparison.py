'''
Comparison of PIMUS to IM
'''
from __future__ import division
__author__ = 'sivanov'
import random

def run_ic(G, S, R):
    spread = 0
    for _ in range(R):
        T = [node for node in S]
        activated = dict(zip(G.nodes(), [False]*len(G)))
        for u in S:
            activated[u] = True
        i = 0
        while i < len(T):
            u = T[i]
            for v in G[u]:
                if not activated[v] and random.random() < G[u][v]['weight']:
                    activated[v] = True
                    T.append(v)
            i += 1
        spread += len(T)
    return spread/R

def run_lt(G, S, R):
    spread = 0

    for _ in range(R):
        thetas = dict(zip(G.nodes(), [random.uniform(0,1) for i in range(len(G))]))
        activated = dict(zip(G.nodes(), [False]*len(G)))
        activated.update(zip(S, [True]*len(S)))
        T = [node for node in S]
        count = len(S)
        while T:
            neighbors = set([v for node in T for v in G[node] if not activated[v]])
            T = []
            for u in neighbors:
                if sum([G[v][u]['weight'] for (v,_) in G.in_edges(u) if activated[v]]) > thetas[u]:
                    activated[u] = True
                    T.append(u)
                    count += 1
        spread += count
    return spread/R


def max_node(G, S, R, IC):
    max_u = None
    max_spread = 0
    for u in G:
        if IC:
            spread = run_ic(G, S + [u], R)
        else:
            spread = run_lt(G, S + [u], R)
        if spread > max_spread:
            max_u = u
            max_spread = spread
    assert max_u is not None
    return max_u

def greedy_im(G, k, R, IC=True):
    S = []
    while len(S) < k:
        node = max_node(G, S, R, IC)
        S.append(node)
    return S


if __name__ == "__main__":

    console = []
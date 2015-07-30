'''

'''
from __future__ import division
import random

__author__ = 'sivanov'


def run_ic(G, L, S, features, R):
    spread = 0
    for _ in range(R):
        activated = dict(zip(G.nodes(), [False]*len(G)))
        for node in S:
            activated[node] = True
        T = [node for node in S]
        i = 0
        while i < len(T):
            v = T[i]
            for neighbor in G[v]:
                if not activated[neighbor]:
                    kappa = len(set(L[neighbor]).intersection(features))/len(L[neighbor])
                    prob = G[v][neighbor]['weight']*kappa
                    if random.random() < prob:
                        activated[neighbor] = True
                        T.append(neighbor)
            i += 1
        spread += len(T)
    return spread/R


def run_lt(G, L, S, features, R):
    spread = 0
    for _ in range(R):
        thetas = dict(zip(G.nodes(), [random.uniform(0, 1) for _ in range(len(G))]))
        activated = dict(zip(G.nodes(), [False]*len(G)))
        count = 0
        for node in S:
            activated[node] = True
            count += 1
        T = [node for node in S]
        while T:
            neighbors = set()
            for node in T:
                for (_, a) in G.out_edges(node):
                    if not activated[a]:
                        neighbors.add(a)
            T = []
            for u in neighbors:
                s = sum([G[a][u]['weight']*len(set(L[u]).intersection(features))/len(L[u]) for (a, _) in G.in_edges(u) if activated[a]])
                if s > thetas[u]:
                    activated[u] = True
                    T.append(u)
                    count += 1
        spread += count
    return spread/R

if __name__ == "__main__":


    console = []
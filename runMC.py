'''

'''
from __future__ import division
import random

def run_mc(G, L, S, features, R):
    spread = 0
    for _ in range(R):
        activated = dict(zip(G.nodes(), [False]*len(G)))
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

__author__ = 'sivanov'

if __name__ == "__main__":
    console = []
'''

'''
from __future__ import division
import random, itertools

__author__ = 'sivanov'


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


def max_feat(G, L, S, features, R, F):
    best_feat_spread = 0
    for feat in F:
        print feat,
        spread = run_mc(G, L, S, features + [feat], R)
        if spread > best_feat_spread:
            best_feat = feat
            best_feat_spread = spread
    print
    assert best_feat
    return best_feat

def greedy(G, L, S, K, R):
    """
    :param G: a graph with probabilities on edges
    :param L: map of a user to its likes
    :param K: total number of selected features
    :param R: number of MC simulations
    :return: features: K selected features
    """
    F = set(list(itertools.chain.from_iterable(L.values())))
    print F
    features = []
    while len(features) < K:
        print len(features) + 1,
        feat = max_feat(G, L, S, features, R, F)
        print 'Selected feature', feat
        F.remove(feat)
        assert feat not in features
        features.append(feat)
    return features

if __name__ == "__main__":
    console = []
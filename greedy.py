'''

'''
from __future__ import division
import itertools
from runMC import run_mc
__author__ = 'sivanov'


def max_feat(G, L, S, features, R, F):
    best_feat_spread = 0
    for feat in F:
        spread = run_mc(G, L, S, features + [feat], R)
        if spread > best_feat_spread:
            best_feat = feat
            best_feat_spread = spread
    assert best_feat
    return best_feat


def greedy(G, L, S, K, R):
    """
    :param G: a graph with probabilities on edges
    :param L: map of a user to its likes
    :param S:
    :param K: total number of selected features
    :param R: number of MC simulations
    :return: features: K selected features
    """
    print 'Starting greedy selection...'
    F = set(list(itertools.chain.from_iterable(L.values())))
    features = []
    print 'Selected nodes: ',
    while len(features) < K:
        feat = max_feat(G, L, S, features, R, F)
        F.remove(feat)
        assert feat not in features
        features.append(feat)
        print feat,
    print
    return features

if __name__ == "__main__":
    console = []